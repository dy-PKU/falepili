from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover
    curl_requests = None

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None

from .common import (
    append_jsonl,
    ensure_dirs,
    load_context,
    normalize_text,
    parse_date,
    read_json,
    save_table,
    setup_logging,
    utc_now_iso,
    write_json,
)


class RedditForbiddenError(RuntimeError):
    """Reddit rejected the request before returning JSON."""


class RedditOAuthError(RuntimeError):
    """Reddit OAuth token or request error."""


class RedditApisError(RuntimeError):
    """RedditAPIs.com request or auth error."""


STRICT_RELEVANCE_POLICY = "reddit_strict_v1"
POST_RELEVANCE_WINDOW_CHARS = 1200
RELEVANT_TOPIC_PATTERNS = (
    re.compile(r"falepili", re.IGNORECASE),
    re.compile(
        r"(tuvalu.{0,120}(australia|australian).{0,120}(visa|visas|migrat|mobility|residenc|treaty|union|agreement|deal|relocat|citizen|pathway|climate))|"
        r"((australia|australian).{0,120}tuvalu.{0,120}(visa|visas|migrat|mobility|residenc|treaty|union|agreement|deal|relocat|citizen|pathway|climate))",
        re.IGNORECASE,
    ),
    re.compile(
        r"tuvalu.{0,120}(climate change|climate visa|climate visas|climate migration|climate refugee|climate refugees|sea level|rising seas|sinking|uninhabitable|new home)",
        re.IGNORECASE,
    ),
)
OFF_TOPIC_PATTERNS = (
    re.compile(r"\b(wts|wtt|wttf|bin|proof pictures|coin conditions|no minimum purchase|shipping method)\b", re.IGNORECASE),
    re.compile(r"\b(pokemon|pokestops|postcards|travel insurance|oneworld|avios)\b", re.IGNORECASE),
    re.compile(r"\b(reference data|countries and territories in scope|practical reference list)\b", re.IGNORECASE),
)


class RedditOAuthClient:
    def __init__(
        self,
        ctx,
        config: dict[str, Any],
        session: Any,
        logger,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.session = session
        self.logger = logger
        self.base_url = str(
            os.getenv("REDDIT_OAUTH_BASE_URL")
            or config.get("oauth_base_url")
            or config.get("base_url")
            or "https://oauth.reddit.com"
        )
        self.token_url = str(
            os.getenv("REDDIT_TOKEN_URL")
            or config.get("token_url")
            or "https://www.reddit.com/api/v1/access_token"
        )
        self.client_id = os.getenv("REDDIT_CLIENT_ID") or config.get("client_id")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET") or config.get("client_secret")
        self.static_access_token = os.getenv("REDDIT_ACCESS_TOKEN") or config.get("access_token")
        self.refresh_token = os.getenv("REDDIT_REFRESH_TOKEN") or config.get("refresh_token")
        self.username = os.getenv("REDDIT_USERNAME") or config.get("username")
        self.password = os.getenv("REDDIT_PASSWORD") or config.get("password")
        self.grant_type = str(
            os.getenv("REDDIT_GRANT_TYPE")
            or config.get("grant_type")
            or ("refresh_token" if self.refresh_token else "password" if self.username and self.password else "client_credentials")
        ).strip().lower()
        self.timeout = int(config.get("timeout_seconds", 30))
        self.max_retries = int(config.get("max_retries", 5))
        self.user_agent = config.get("user_agent", "falepili-research-test/0.1")
        self.token_cache_path = ctx.paths.interim / "reddit_oauth_token.json"
        self._token_payload: dict[str, Any] | None = None

    def _load_cached_token(self) -> dict[str, Any] | None:
        cached = read_json(self.token_cache_path, default=None)
        if not isinstance(cached, dict):
            return None
        expires_at = float(cached.get("expires_at") or 0)
        if cached.get("access_token") and expires_at > time.time() + 60:
            return cached
        return None

    def _save_token(self, payload: dict[str, Any]) -> None:
        write_json(self.token_cache_path, payload)

    def _basic_auth_header(self) -> str:
        if not self.client_id:
            raise RedditOAuthError("Missing Reddit client_id. Set REDDIT_CLIENT_ID or config.reddit_json.client_id.")
        secret = self.client_secret or ""
        token = base64.b64encode(f"{self.client_id}:{secret}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _build_token_request(self) -> dict[str, Any]:
        if self.grant_type == "client_credentials":
            if not self.client_id:
                raise RedditOAuthError(
                    "Missing REDDIT_CLIENT_ID for client_credentials grant."
                )
            return {"grant_type": "client_credentials"}
        if self.grant_type == "refresh_token":
            if not self.refresh_token:
                raise RedditOAuthError(
                    "Missing REDDIT_REFRESH_TOKEN for refresh_token grant."
                )
            return {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        if self.grant_type == "password":
            if not self.username or not self.password:
                raise RedditOAuthError(
                    "Missing REDDIT_USERNAME/REDDIT_PASSWORD for password grant."
                )
            return {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            }
        raise RedditOAuthError(f"Unsupported Reddit OAuth grant_type: {self.grant_type}")

    def access_token(self, force_refresh: bool = False) -> str:
        if self.static_access_token:
            return str(self.static_access_token)
        if not force_refresh:
            if self._token_payload is None:
                self._token_payload = self._load_cached_token()
            if self._token_payload:
                return str(self._token_payload["access_token"])
        token_payload = self._request_access_token()
        self._token_payload = token_payload
        return str(token_payload["access_token"])

    def _request_access_token(self) -> dict[str, Any]:
        data = self._build_token_request()
        headers = {
            "Authorization": self._basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self.user_agent,
        }
        response = self.session.post(
            self.token_url,
            data=data,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RedditOAuthError(
                f"Reddit token request failed: {response.status_code} {response.text[:1000]}"
            )
        payload = response.json()
        if "access_token" not in payload:
            raise RedditOAuthError(f"Reddit token response missing access_token: {payload}")
        expires_in = int(payload.get("expires_in") or 3600)
        payload["expires_at"] = time.time() + expires_in
        self._save_token(payload)
        return payload

    def request_json(self, url: str, params: dict[str, Any], logger, timeout: int | None = None) -> dict[str, Any]:
        timeout = timeout or self.timeout
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            headers = {
                "Authorization": f"bearer {self.access_token(force_refresh=attempt > 0)}",
                "User-Agent": self.user_agent,
            }
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=timeout)
                if response.status_code == 401:
                    self._token_payload = None
                    raise RedditOAuthError("Reddit OAuth token rejected with 401.")
                if response.status_code == 403:
                    body = response.text[:1000] if response.text else ""
                    raise RedditForbiddenError(f"Reddit returned 403 Forbidden. Body preview: {body}")
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait = float(response.headers.get("Retry-After") or min(90, 2**attempt + random.random()))
                    logger.warning("reddit_retryable_status status=%s wait=%.1f url=%s", response.status_code, wait, url)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if isinstance(exc, RedditForbiddenError):
                    raise
                last_error = exc
                wait = min(60, 2**attempt + random.random())
                logger.warning("reddit_oauth_request_failed attempt=%s wait=%.1f error=%s", attempt + 1, wait, exc)
                time.sleep(wait)
        raise RedditOAuthError(f"Reddit OAuth request failed after {self.max_retries} attempts: {last_error}")


class RedditApisClient:
    def __init__(
        self,
        ctx,
        config: dict[str, Any],
        session: Any,
        logger,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.session = session
        self.logger = logger
        self.base_url = str(
            os.getenv("REDDITAPIS_BASE_URL")
            or config.get("redditapis_base_url")
            or config.get("api_base_url")
            or "https://api.redditapis.com"
        )
        self.timeout = int(config.get("timeout_seconds", 30))
        self.max_retries = int(config.get("max_retries", 5))
        self.user_agent = config.get("user_agent", "falepili-research-test/0.1")
        self.api_key = self._load_api_key(config)
        if not self.api_key:
            raise RedditApisError(
                "Missing RedditAPIs key. Set REDDITAPI_KEY or point reddit_json.api_key_path at key.txt."
            )

    def _load_api_key(self, config: dict[str, Any]) -> str | None:
        env_key = os.getenv("REDDITAPI_KEY") or os.getenv("REDDIT_APIS_KEY")
        if env_key:
            return env_key.strip()
        key_path_raw = (
            os.getenv("REDDITAPI_KEY_PATH")
            or os.getenv("REDDIT_APIS_KEY_PATH")
            or config.get("api_key_path")
            or config.get("redditapis_key_path")
            or "key.txt"
        )
        key_path = Path(str(key_path_raw).strip())
        candidates = []
        if key_path.is_absolute():
            candidates.append(key_path)
        else:
            candidates.extend(
                [
                    ctx_candidate
                    for ctx_candidate in (
                        self.ctx.paths.root / key_path,
                        self.ctx.paths.root.parent / key_path,
                        Path.cwd() / key_path,
                    )
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                try:
                    for line in candidate.read_text(encoding="utf-8").splitlines():
                        token = line.strip()
                        if token:
                            return token
                except OSError:
                    continue
        return None

    def request_json(self, url: str, params: dict[str, Any], logger, timeout: int | None = None) -> dict[str, Any]:
        timeout = timeout or self.timeout
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.user_agent,
            }
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=timeout)
                if response.status_code == 403:
                    body = response.text[:1000] if response.text else ""
                    raise RedditForbiddenError(f"RedditAPIs returned 403 Forbidden. Body preview: {body}")
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait = float(response.headers.get("Retry-After") or min(90, 2**attempt + random.random()))
                    logger.warning("redditapis_retryable_status status=%s wait=%.1f url=%s", response.status_code, wait, url)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if isinstance(exc, RedditForbiddenError):
                    raise
                last_error = exc
                wait = min(60, 2**attempt + random.random())
                logger.warning("redditapis_request_failed attempt=%s wait=%.1f error=%s", attempt + 1, wait, exc)
                time.sleep(wait)
        raise RedditApisError(f"RedditAPIs request failed after {self.max_retries} attempts: {last_error}")


def _load_reddit_config(ctx) -> dict[str, Any]:
    path = ctx.paths.config / "reddit.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("reddit_json", {})


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _auth_mode(config: dict[str, Any]) -> str:
    return str(
        os.getenv("REDDIT_AUTH_MODE")
        or config.get("auth_mode")
        or ("oauth" if config.get("client_id") or os.getenv("REDDIT_CLIENT_ID") else "public")
    ).strip().lower()


def _redditapis_base_url(config: dict[str, Any]) -> str:
    return str(
        os.getenv("REDDITAPIS_BASE_URL")
        or config.get("redditapis_base_url")
        or config.get("api_base_url")
        or "https://api.redditapis.com"
    )


def _redditapis_search_url(base_url: str, subreddit: str | None) -> str:
    return f"{base_url.rstrip('/')}/api/reddit/search"


def _redditapis_comments_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/reddit/comments"


def _redditapis_search_params(keyword: str, subreddit: str | None, after: str | None, limit: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": keyword,
        "sort": "new",
        "time_filter": "all",
        "limit": limit,
    }
    if subreddit:
        params["subreddit"] = subreddit
    if after:
        params["after"] = after
    return params


def _redditapis_comment_params(permalink: str, depth: int = 10, limit: int = 500, sort: str = "confidence") -> dict[str, Any]:
    return {
        "permalink": permalink,
        "depth": depth,
        "limit": limit,
        "sort": sort,
    }


def _relevance_text(text: Any) -> str:
    return normalize_text(text).lower()


def _is_direct_post_relevant(text: Any) -> bool:
    text_l = _relevance_text(text)[:POST_RELEVANCE_WINDOW_CHARS]
    if not text_l:
        return False
    if any(pattern.search(text_l) for pattern in OFF_TOPIC_PATTERNS):
        return False
    return any(pattern.search(text_l) for pattern in RELEVANT_TOPIC_PATTERNS)


def _is_direct_comment_relevant(text: Any) -> bool:
    text_l = _relevance_text(text)
    if not text_l:
        return False
    if any(pattern.search(text_l) for pattern in OFF_TOPIC_PATTERNS):
        return False
    direct_hints = (
        "falepili union",
        "falepili mobility pathway",
        "australia-tuvalu",
        "tuvalu australia treaty",
        "tuvalu climate migration",
        "tuvalu climate refugees",
        "tuvalu permanent residency",
    )
    if any(hint in text_l for hint in direct_hints):
        return True
    return any(pattern.search(text_l) for pattern in RELEVANT_TOPIC_PATTERNS)


def _comment_text(value: dict[str, Any]) -> str:
    return normalize_text(value.get("body") or value.get("text") or "")


def _date_bounds(config: dict[str, Any]) -> tuple[int | None, int | None]:
    start = parse_date(config.get("start_date"))
    end = parse_date(config.get("end_date")) or dt.datetime.now(tz=dt.timezone.utc).date()
    start_ts = int(dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc).timestamp()) if start else None
    end_ts = int(dt.datetime.combine(end, dt.time.max, tzinfo=dt.timezone.utc).timestamp()) if end else None
    return start_ts, end_ts


def _session(proxy_url: str | None, use_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_env_proxy
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session


def _curl_session(proxy_url: str | None, use_env_proxy: bool, impersonate: str) -> Any:
    if curl_requests is None:
        raise RuntimeError(
            "curl_cffi is not installed. Install requirements to enable TLS impersonation."
        )
    session = curl_requests.Session(impersonate=impersonate, trust_env=use_env_proxy)
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session


def build_http_session(config: dict[str, Any], use_env_proxy: bool, proxy_url: str | None) -> Any:
    transport = str(config.get("transport", "requests")).strip().lower()
    impersonate = str(config.get("impersonate", "chrome")).strip()
    if transport == "curl_cffi":
        return _curl_session(proxy_url=proxy_url, use_env_proxy=use_env_proxy, impersonate=impersonate)
    return _session(proxy_url=proxy_url, use_env_proxy=use_env_proxy)


def _browser_enabled(config: dict[str, Any]) -> bool:
    mode = str(
        os.getenv("REDDIT_BROWSER_MODE")
        or config.get("browser_mode")
        or "off"
    ).strip().lower()
    return mode in {"playwright", "browser", "on", "true", "1"}


def _browser_channel(config: dict[str, Any]) -> str:
    return str(os.getenv("REDDIT_BROWSER_CHANNEL") or config.get("browser_channel") or "chrome")


def _browser_executable(config: dict[str, Any]) -> str | None:
    return os.getenv("REDDIT_BROWSER_EXECUTABLE") or config.get("browser_executable")


def _browser_proxy(proxy_url: str | None, use_env_proxy: bool, config: dict[str, Any]) -> str | None:
    return proxy_url or os.getenv("REDDIT_PROXY_URL") or config.get("proxy_url") or (os.environ.get("HTTPS_PROXY") if use_env_proxy else None)


def _browser_context_kwargs(proxy_url: str | None, use_env_proxy: bool, config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    proxy_value = _browser_proxy(proxy_url, use_env_proxy, config)
    if proxy_value:
        kwargs["proxy"] = {"server": proxy_value}
    return kwargs


def _browser_search_query(keyword: str, subreddit: str | None) -> str:
    if subreddit:
        return f"site:reddit.com/r/{subreddit} {keyword}"
    return f"site:reddit.com {keyword}"


def _browser_fetch_posts(
    ctx,
    config: dict[str, Any],
    keywords: list[str],
    subreddits: list[str | None],
    max_pages: int,
    proxy_url: str | None,
    use_env_proxy: bool,
) -> list[dict[str, Any]]:
    if sync_playwright is None:
        raise RuntimeError("playwright is not installed. Install requirements to enable browser mode.")
    logger = setup_logging("reddit_json")
    base_url = "https://www.reddit.com"
    limit = int(config.get("search_limit", 100))
    start_ts, end_ts = _date_bounds(config)
    channel = _browser_channel(config)
    # Resume safely: completed jobs are skipped, so retain previously normalized
    # rows and merge newly fetched subreddit/query jobs into the same dataset.
    existing_posts_path = ctx.paths.processed / "reddit_posts.parquet"
    if existing_posts_path.exists():
        existing_posts = pd.read_parquet(existing_posts_path)
        posts: list[dict[str, Any]] = existing_posts.to_dict(orient="records")
    else:
        posts = []
    with sync_playwright() as p:
        browser_type = getattr(p, channel, None)
        if browser_type is None:
            browser_type = p.chromium
        launch_kwargs: dict[str, Any] = {"headless": True}
        executable = _browser_executable(config)
        if executable:
            launch_kwargs["executable_path"] = executable
        browser_context_kwargs = _browser_context_kwargs(proxy_url, use_env_proxy, config)
        browser = browser_type.launch(**launch_kwargs)
        context = browser.new_context(**browser_context_kwargs)
        page = context.new_page()
        for subreddit in subreddits:
            for keyword in keywords:
                query = _browser_search_query(keyword, subreddit)
                url = f"{base_url}/search/?q={requests.utils.quote(query)}&sort=new"
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(2000)
                except Exception as exc:
                    logger.warning("browser_search_failed keyword=%s subreddit=%s error=%s", keyword, subreddit, exc)
                    continue
                retrieved_at = utc_now_iso()
                append_jsonl(
                    ctx.paths.raw / "reddit_browser_search_responses.jsonl",
                    {
                        "retrieved_at": retrieved_at,
                        "keyword": keyword,
                        "subreddit": subreddit or "all",
                        "url": url,
                        "source": "browser",
                    },
                )
                cards = page.locator('shreddit-post')
                count = min(cards.count(), max_pages * limit)
                for idx in range(count):
                    card = cards.nth(idx)
                    try:
                        post_id = card.get_attribute("id") or card.get_attribute("permalink") or card.get_attribute("name")
                        title = card.locator("a[href*='/comments/']").first.inner_text(timeout=3000)
                        href = card.locator("a[href*='/comments/']").first.get_attribute("href")
                    except Exception:
                        continue
                    if not post_id and href:
                        post_id = href.split("/comments/")[-1].strip("/").split("/")[0]
                    if not post_id:
                        continue
                    title = normalize_text(title)
                    if not title:
                        continue
                    if not _is_direct_post_relevant(title):
                        continue
                    posts.append(
                        {
                            "reddit_post_id": post_id,
                            "fullname": None,
                            "subreddit": subreddit or None,
                            "subreddit_scope": subreddit or "all",
                            "title": title,
                            "selftext": "",
                            "url": href,
                            "permalink": href,
                            "author": None,
                            "created_utc": None,
                            "created_at": None,
                            "score": None,
                            "upvote_ratio": None,
                            "num_comments": None,
                            "over_18": None,
                            "stickied": None,
                            "locked": None,
                            "query_source": keyword,
                            "retrieved_at": retrieved_at,
                            "source": "reddit_browser",
                            "relevance_status": "directly relevant",
                            "relevance_score": 1,
                        }
                    )
        context.close()
        browser.close()
    return posts


def _redditapis_post_from_item(item: dict[str, Any], keyword: str, subreddit_scope: str | None, retrieved_at: str) -> dict[str, Any]:
    created_utc = item.get("created_utc")
    selftext = item.get("selftext") or item.get("text") or item.get("body") or ""
    score = item.get("upvotes")
    if score is None:
        score = item.get("score")
    num_comments = item.get("comments")
    if num_comments is None:
        num_comments = item.get("num_comments")
    return {
        "reddit_post_id": item.get("id"),
        "fullname": item.get("name"),
        "subreddit": item.get("subreddit"),
        "subreddit_scope": subreddit_scope or "all",
        "title": normalize_text(item.get("title")),
        "selftext": normalize_text(selftext),
        "url": item.get("url"),
        "permalink": item.get("permalink"),
        "author": item.get("author"),
        "created_utc": created_utc,
        "created_at": dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z") if created_utc else None,
        "score": score,
        "upvote_ratio": item.get("upvote_ratio"),
        "num_comments": num_comments,
        "over_18": item.get("over_18"),
        "stickied": item.get("stickied"),
        "locked": item.get("locked"),
        "query_source": keyword,
        "retrieved_at": retrieved_at,
        "source": "redditapis_search",
        "relevance_status": "directly relevant" if _is_direct_post_relevant(f"{item.get('title') or ''} {selftext}") else "irrelevant",
        "relevance_score": 1 if _is_direct_post_relevant(f"{item.get('title') or ''} {selftext}") else 0,
    }


def _comment_records_from_nodes(
    post_id: str,
    nodes: list[dict[str, Any]] | None,
    retrieved_at: str,
    parent_id: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for child in nodes or []:
        kind = child.get("kind")
        data = child.get("data", {}) if isinstance(child, dict) else {}
        if kind == "more":
            records.append(
                {
                    "reddit_comment_id": data.get("id"),
                    "reddit_post_id": post_id,
                    "parent_id": parent_id,
                    "body": "",
                    "author": None,
                    "created_utc": None,
                    "created_at": None,
                    "score": None,
                    "is_submitter": None,
                    "depth": depth,
                    "retrieved_at": retrieved_at,
                    "source": "reddit_json_more_placeholder",
                    "more_count": data.get("count"),
                    "relevance_status": "irrelevant",
                    "relevance_score": 0,
                }
            )
            continue
        if kind != "t1":
            continue
        created_utc = data.get("created_utc")
        comment_id = data.get("id")
        records.append(
            {
                "reddit_comment_id": comment_id,
                "reddit_post_id": post_id,
                "parent_id": data.get("parent_id") or parent_id,
                "body": normalize_text(data.get("body")),
                "author": data.get("author"),
                "created_utc": created_utc,
                "created_at": dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z") if created_utc else None,
                "score": data.get("score"),
                "is_submitter": data.get("is_submitter"),
                "depth": depth,
                "retrieved_at": retrieved_at,
                "source": "reddit_json_comments",
                "more_count": None,
                "relevance_status": "directly relevant" if _is_direct_comment_relevant(data.get("body")) else "irrelevant",
                "relevance_score": 1 if _is_direct_comment_relevant(data.get("body")) else 0,
            }
        )
        replies = data.get("replies")
        if isinstance(replies, dict):
            nested = replies.get("data", {}).get("children", [])
            records.extend(_comment_records_from_nodes(post_id, nested, retrieved_at, parent_id=comment_id, depth=depth + 1))
    return records


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    logger,
    timeout: int,
    max_retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 403:
                body = response.text[:1000] if response.text else ""
                raise RedditForbiddenError(f"Reddit returned 403 Forbidden. Body preview: {body}")
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = float(response.headers.get("Retry-After") or min(90, 2**attempt + random.random()))
                logger.warning("reddit_retryable_status status=%s wait=%.1f url=%s", response.status_code, wait, url)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if isinstance(exc, RedditForbiddenError):
                raise
            last_error = exc
            wait = min(60, 2**attempt + random.random())
            logger.warning("reddit_request_failed attempt=%s wait=%.1f error=%s", attempt + 1, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"Reddit request failed after {max_retries} attempts: {last_error}")


def _search_url(base_url: str, subreddit: str | None, oauth_mode: bool = False) -> str:
    suffix = "search" if oauth_mode else "search.json"
    if subreddit:
        return f"{base_url.rstrip('/')}/r/{subreddit}/{suffix}"
    return f"{base_url.rstrip('/')}/{suffix}"


def _comments_url(base_url: str, subreddit: str | None, article_id: str, oauth_mode: bool = False) -> str:
    suffix = "" if oauth_mode else ".json"
    if subreddit:
        return f"{base_url.rstrip('/')}/r/{subreddit}/comments/{article_id}{suffix}"
    return f"{base_url.rstrip('/')}/comments/{article_id}{suffix}"


def _morechildren_url(base_url: str, oauth_mode: bool = False) -> str:
    suffix = "" if oauth_mode else ".json"
    return f"{base_url.rstrip('/')}/api/morechildren{suffix}"


def _post_from_child(child: dict[str, Any], keyword: str, subreddit_scope: str | None, retrieved_at: str) -> dict[str, Any]:
    data = child.get("data", {})
    created_utc = data.get("created_utc")
    return {
        "reddit_post_id": data.get("id"),
        "fullname": data.get("name"),
        "subreddit": data.get("subreddit"),
        "subreddit_scope": subreddit_scope or "all",
        "title": normalize_text(data.get("title")),
        "selftext": normalize_text(data.get("selftext")),
        "url": data.get("url"),
        "permalink": data.get("permalink"),
        "author": data.get("author"),
        "created_utc": created_utc,
        "created_at": dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z") if created_utc else None,
        "score": data.get("score"),
        "upvote_ratio": data.get("upvote_ratio"),
        "num_comments": data.get("num_comments"),
        "over_18": data.get("over_18"),
        "stickied": data.get("stickied"),
        "locked": data.get("locked"),
        "query_source": keyword,
        "retrieved_at": retrieved_at,
        "source": "reddit_json_search",
    }


def _comment_records_from_listing(
    post_id: str,
    listing: dict[str, Any],
    retrieved_at: str,
    parent_id: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    children = listing.get("data", {}).get("children", []) or []
    return _comment_records_from_nodes(post_id, children, retrieved_at, parent_id=parent_id, depth=depth)


def search_posts(
    ctx,
    max_pages_per_query: int | None = None,
    use_env_proxy: bool = True,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    config = _load_reddit_config(ctx)
    logger = setup_logging("reddit_json")
    ensure_dirs(ctx.paths)
    auth_mode = _auth_mode(config)
    browser_mode = _browser_enabled(config) and auth_mode != "redditapis"
    if auth_mode == "oauth":
        base_url = str(
            os.getenv("REDDIT_OAUTH_BASE_URL")
            or config.get("oauth_base_url")
            or "https://oauth.reddit.com"
        )
    elif auth_mode == "redditapis":
        base_url = _redditapis_base_url(config)
    else:
        base_url = str(config.get("base_url", "https://www.reddit.com"))
    keywords = config.get("keywords") or ctx.keywords_config.get("keywords", [])
    subreddits = config.get("subreddits") or [None]
    limit = int(config.get("search_limit", 100))
    timeout = int(config.get("timeout_seconds", 30))
    max_retries = int(config.get("max_retries", 5))
    delay = float(config.get("request_delay_seconds", 2.0))
    max_pages = int(max_pages_per_query or config.get("default_max_pages_per_query", 5))
    strict_relevance = _env_bool("REDDIT_STRICT_RELEVANCE", bool(config.get("strict_relevance", True)))
    strict_comment_relevance = _env_bool(
        "REDDIT_STRICT_COMMENT_RELEVANCE",
        bool(config.get("strict_comment_relevance", False)),
    )
    start_ts, end_ts = _date_bounds(config)
    headers = {"User-Agent": config.get("user_agent", "falepili-research-test/0.1")}
    proxy_url = proxy_url or os.getenv("REDDIT_PROXY_URL") or config.get("proxy_url")
    if proxy_url is None and not use_env_proxy:
        use_env_proxy = False
    session = build_http_session(config, use_env_proxy=use_env_proxy, proxy_url=proxy_url)
    redditapis_client = RedditApisClient(ctx, config, session, logger) if auth_mode == "redditapis" else None
    oauth_client = RedditOAuthClient(ctx, config, session, logger) if auth_mode == "oauth" else None
    if browser_mode:
        keywords = config.get("keywords") or ctx.keywords_config.get("keywords", [])
        subreddits = config.get("subreddits") or [None]
        posts = _browser_fetch_posts(
            ctx=ctx,
            config=config,
            keywords=keywords,
            subreddits=subreddits,
            max_pages=int(max_pages_per_query or config.get("default_max_pages_per_query", 5)),
            proxy_url=proxy_url,
            use_env_proxy=use_env_proxy,
        )
        df = pd.DataFrame(posts)
        if not df.empty:
            df = df.drop_duplicates(subset=["reddit_post_id"], keep="last")
            save_table(df, ctx.paths.processed / "reddit_posts.parquet")
        else:
            save_table(pd.DataFrame(columns=["reddit_post_id"]), ctx.paths.processed / "reddit_posts.parquet")
        return df
    state_path = ctx.paths.interim / "reddit_search_state.json"
    state = read_json(state_path, default={"jobs": {}})
    if strict_relevance and state.get("policy") != STRICT_RELEVANCE_POLICY:
        state = {"policy": STRICT_RELEVANCE_POLICY, "jobs": {}}
    posts: list[dict[str, Any]] = []

    for subreddit in subreddits:
        for keyword in keywords:
            job_key = f"{subreddit or 'all'}::{keyword}"
            job = state.setdefault("jobs", {}).setdefault(job_key, {"after": None, "pages_done": 0, "finished": False})
            if job.get("finished"):
                logger.info("reddit_search_skip job=%s", job_key)
                continue
            after = job.get("after")
            for page in range(int(job.get("pages_done", 0)), max_pages):
                retrieved_at = utc_now_iso()
                if redditapis_client:
                    params = _redditapis_search_params(keyword, subreddit, after, limit)
                    url = _redditapis_search_url(base_url, subreddit)
                    payload = redditapis_client.request_json(url, params, logger, timeout=timeout)
                else:
                    params = {
                        "q": keyword,
                        "sort": "new",
                        "t": "all",
                        "limit": limit,
                        "raw_json": 1,
                    }
                    if subreddit:
                        params["restrict_sr"] = "on"
                    if after:
                        params["after"] = after
                    url = _search_url(base_url, subreddit, oauth_mode=oauth_client is not None)
                    payload = (
                        oauth_client.request_json(url, params, logger, timeout=timeout)
                        if oauth_client
                        else _request_json(session, url, params, headers, logger, timeout, max_retries)
                    )
                append_jsonl(
                    ctx.paths.raw / "reddit_search_responses.jsonl",
                    {
                        "retrieved_at": retrieved_at,
                        "keyword": keyword,
                        "subreddit": subreddit or "all",
                        "page": page,
                        "after": after,
                        "url": url,
                        "params": params,
                        "payload": payload,
                    },
                )
                if redditapis_client:
                    items = payload.get("posts", []) if isinstance(payload, dict) else []
                    rows = [_redditapis_post_from_item(item, keyword, subreddit, retrieved_at) for item in items]
                    after = payload.get("after") if isinstance(payload, dict) else None
                else:
                    children = payload.get("data", {}).get("children", []) if isinstance(payload, dict) else []
                    rows = [_post_from_child(child, keyword, subreddit, retrieved_at) for child in children]
                    after = payload.get("data", {}).get("after") if isinstance(payload, dict) else None
                for row in rows:
                    created_utc = row.get("created_utc")
                    if start_ts and created_utc and created_utc < start_ts:
                        continue
                    if end_ts and created_utc and created_utc > end_ts:
                        continue
                    is_relevant = _is_direct_post_relevant(f"{row.get('title') or ''} {row.get('selftext') or ''}")
                    row["relevance_status"] = "directly relevant" if is_relevant else "irrelevant"
                    row["relevance_score"] = 1 if is_relevant else 0
                    if strict_relevance and not is_relevant:
                        continue
                    posts.append(row)
                job["after"] = after
                job["pages_done"] = page + 1
                if not after:
                    job["finished"] = True
                write_json(state_path, state)
                logger.info("reddit_search_page job=%s page=%s posts_seen=%s after=%s", job_key, page + 1, len(rows), after)
                time.sleep(delay)
                if not after:
                    break

    df = pd.DataFrame(posts)
    if not df.empty:
        df = df.drop_duplicates(subset=["reddit_post_id"], keep="last")
        save_table(df, ctx.paths.processed / "reddit_posts.parquet")
    else:
        save_table(pd.DataFrame(columns=["reddit_post_id"]), ctx.paths.processed / "reddit_posts.parquet")
    logger.info("reddit_search_complete posts=%s", len(df))
    return df


def fetch_comments(
    ctx,
    posts: pd.DataFrame | None = None,
    max_posts: int | None = None,
    use_env_proxy: bool = True,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    config = _load_reddit_config(ctx)
    logger = setup_logging("reddit_json")
    ensure_dirs(ctx.paths)
    auth_mode = _auth_mode(config)
    browser_mode = _browser_enabled(config) and auth_mode != "redditapis"
    if auth_mode == "oauth":
        base_url = str(
            os.getenv("REDDIT_OAUTH_BASE_URL")
            or config.get("oauth_base_url")
            or "https://oauth.reddit.com"
        )
    elif auth_mode == "redditapis":
        base_url = _redditapis_base_url(config)
    else:
        base_url = str(config.get("base_url", "https://www.reddit.com"))
    timeout = int(config.get("timeout_seconds", 30))
    max_retries = int(config.get("max_retries", 5))
    delay = float(config.get("request_delay_seconds", 2.0))
    headers = {"User-Agent": config.get("user_agent", "falepili-research-test/0.1")}
    proxy_url = proxy_url or os.getenv("REDDIT_PROXY_URL") or config.get("proxy_url")
    session = build_http_session(config, use_env_proxy=use_env_proxy, proxy_url=proxy_url)
    strict_relevance = _env_bool("REDDIT_STRICT_RELEVANCE", bool(config.get("strict_relevance", True)))
    redditapis_client = RedditApisClient(ctx, config, session, logger) if auth_mode == "redditapis" else None
    oauth_client = RedditOAuthClient(ctx, config, session, logger) if auth_mode == "oauth" else None
    if browser_mode:
        save_table(pd.DataFrame(columns=["reddit_comment_id"]), ctx.paths.processed / "reddit_comments.parquet")
        return pd.DataFrame(columns=["reddit_comment_id"])
    posts_path = ctx.paths.processed / "reddit_posts.parquet"
    if posts is None:
        posts = pd.read_parquet(posts_path) if posts_path.exists() else pd.DataFrame()
    if posts.empty:
        save_table(pd.DataFrame(columns=["reddit_comment_id"]), ctx.paths.processed / "reddit_comments.parquet")
        return pd.DataFrame()
    max_posts = int(max_posts or config.get("max_posts_for_comments", 200))
    # Preserve comments from completed posts when a later run only fetches new
    # candidates; otherwise the resume state could cause an empty overwrite.
    comments_path = ctx.paths.processed / "reddit_comments.parquet"
    if comments_path.exists():
        existing_comments = pd.read_parquet(comments_path)
        comments: list[dict[str, Any]] = existing_comments.to_dict(orient="records")
    else:
        comments = []
    state_path = ctx.paths.interim / "reddit_comments_state.json"
    state = read_json(state_path, default={"posts": {}})
    if strict_relevance and state.get("policy") != STRICT_RELEVANCE_POLICY:
        state = {"policy": STRICT_RELEVANCE_POLICY, "posts": {}}
    if strict_relevance:
        posts = posts.copy()
        posts["__keep"] = posts.apply(lambda row: _is_direct_post_relevant(f"{row.get('title') or ''} {row.get('selftext') or ''}"), axis=1)
        posts = posts[posts["__keep"]].drop(columns=["__keep"])
    candidates = posts.sort_values(["num_comments", "score"], ascending=False, na_position="last").head(max_posts)
    for _, post in candidates.iterrows():
        post_id = str(post["reddit_post_id"])
        if state.setdefault("posts", {}).get(post_id, {}).get("finished"):
            logger.info("reddit_comments_skip post_id=%s", post_id)
            continue
        retrieved_at = utc_now_iso()
        if redditapis_client:
            url = _redditapis_comments_url(base_url)
            permalink = str(post.get("permalink") or "")
            params = _redditapis_comment_params(permalink)
            payload = redditapis_client.request_json(url, params, logger, timeout=timeout)
        else:
            url = _comments_url(base_url, None, post_id, oauth_mode=oauth_client is not None)
            params = {"limit": 500, "depth": 10, "sort": "confidence", "raw_json": 1}
            payload = (
                oauth_client.request_json(url, params, logger, timeout=timeout)
                if oauth_client
                else _request_json(session, url, params, headers, logger, timeout, max_retries)
            )
        append_jsonl(
            ctx.paths.raw / "reddit_comment_responses.jsonl",
            {
                "retrieved_at": retrieved_at,
                "post_id": post_id,
                "url": url,
                "params": params,
                "payload": payload,
            },
        )
        if redditapis_client:
            nodes = payload.get("comments", []) if isinstance(payload, dict) else []
            records = _comment_records_from_nodes(post_id, nodes, retrieved_at)
        elif isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], dict):
            records = _comment_records_from_listing(post_id, payload[1], retrieved_at)
        else:
            records = []
        kept_records: list[dict[str, Any]] = []
        for record in records:
            relevant = _is_direct_comment_relevant(record.get("body"))
            record["relevance_status"] = "directly relevant" if relevant else "irrelevant"
            record["relevance_score"] = 1 if relevant else 0
            record["thread_relevance_status"] = "directly relevant"
            if strict_relevance and strict_comment_relevance and not relevant:
                continue
            kept_records.append(record)
        comments.extend(kept_records)
        state["posts"][post_id] = {"finished": True, "completed_at": retrieved_at}
        write_json(state_path, state)
        logger.info("reddit_comments_post post_id=%s comments_total=%s", post_id, len(comments))
        time.sleep(delay)
    df = pd.DataFrame(comments)
    if not df.empty:
        df = df.drop_duplicates(subset=["reddit_comment_id"], keep="last")
        save_table(df, ctx.paths.processed / "reddit_comments.parquet")
    else:
        save_table(pd.DataFrame(columns=["reddit_comment_id"]), ctx.paths.processed / "reddit_comments.parquet")
    logger.info("reddit_comments_complete comments=%s", len(df))
    return df


def run(
    ctx=None,
    max_pages_per_query: int | None = None,
    fetch_comment_tree: bool | None = None,
    max_posts_for_comments: int | None = None,
    use_env_proxy: bool | None = None,
    proxy_url: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = ctx or load_context()
    config = _load_reddit_config(ctx)
    if use_env_proxy is None:
        use_env_proxy = _env_bool("REDDIT_USE_ENV_PROXY", True)
    posts = search_posts(ctx, max_pages_per_query=max_pages_per_query, use_env_proxy=use_env_proxy, proxy_url=proxy_url)
    should_fetch_comments = config.get("fetch_comments", True) if fetch_comment_tree is None else fetch_comment_tree
    comments = (
        fetch_comments(
            ctx,
            posts=posts,
            max_posts=max_posts_for_comments,
            use_env_proxy=use_env_proxy,
            proxy_url=proxy_url,
        )
        if should_fetch_comments
        else pd.DataFrame()
    )
    return posts, comments


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Reddit search results and comment trees through public JSON, Reddit OAuth, or RedditAPIs.com endpoints.")
    parser.add_argument("--max-pages-per-query", type=int, default=None, help="Pages per keyword/subreddit search job.")
    parser.add_argument("--no-comments", action="store_true", help="Collect posts only.")
    parser.add_argument("--max-posts-for-comments", type=int, default=None, help="Maximum posts whose comment trees are fetched.")
    parser.add_argument("--proxy-url", type=str, default=None, help="Explicit proxy URL, e.g. http://127.0.0.1:7890.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Honor HTTP_PROXY/HTTPS_PROXY environment variables.")
    parser.add_argument("--auth-mode", choices=["public", "oauth", "redditapis"], default=None, help="Use public JSON endpoints, Reddit OAuth API, or RedditAPIs.com.")
    args = parser.parse_args()
    if args.auth_mode:
        os.environ["REDDIT_AUTH_MODE"] = args.auth_mode
    run(
        max_pages_per_query=args.max_pages_per_query,
        fetch_comment_tree=not args.no_comments,
        max_posts_for_comments=args.max_posts_for_comments,
        use_env_proxy=args.use_env_proxy or _env_bool("REDDIT_USE_ENV_PROXY", True),
        proxy_url=args.proxy_url,
    )


if __name__ == "__main__":
    main()
