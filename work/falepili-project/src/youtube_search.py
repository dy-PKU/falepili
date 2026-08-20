from __future__ import annotations

import argparse
import datetime as dt
import itertools
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    append_jsonl,
    chunked,
    default_study_dates,
    ensure_dirs,
    iso_datetime_for_date,
    load_context,
    month_windows,
    normalize_text,
    parse_date,
    request_json_with_backoff,
    save_table,
    setup_logging,
    stable_hash,
    utc_now_iso,
    write_jsonl,
)


SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def build_query_jobs(ctx) -> list[dict[str, Any]]:
    keywords = ctx.keywords_config.get("keywords", [])
    start_date, end_date = default_study_dates(ctx)
    jobs: list[dict[str, Any]] = []
    for keyword, (window_start, window_end) in itertools.product(keywords, month_windows(start_date, end_date)):
        jobs.append(
            {
                "keyword": keyword,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start:%Y-%m}",
            }
        )
    return jobs


def _demo_video_rows(ctx, scale: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(ctx.random_seed)
    channels = [
        ("UC1", "ABC News Australia"),
        ("UC2", "SBS News"),
        ("UC3", "Sky News Australia"),
        ("UC4", "7NEWS"),
        ("UC5", "9News"),
        ("UC6", "Guardian Australia"),
    ]
    events = [dt.date.fromisoformat(item["date"]) for item in ctx.events_config.get("events", [])]
    keywords = ctx.keywords_config.get("keywords", [])
    rows: list[dict[str, Any]] = []
    demo_keywords = keywords * max(1, int(scale)) * 2
    for idx, keyword in enumerate(demo_keywords):
        channel_id, channel_name = channels[idx % len(channels)]
        event_date = events[idx % len(events)] if events else dt.date(2023, 11, 9)
        published = event_date + dt.timedelta(days=rng.randint(-35, 180))
        title = f"Australia and Tuvalu: {keyword}"
        description = (
            f"Coverage of the Falepili Union, climate mobility and security cooperation. "
            f"Keyword focus: {keyword}."
        )
        video_id = f"demo_{stable_hash([keyword, idx])[:10]}"
        row = {
            "video_id": video_id,
            "title": title,
            "description": description,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "published_at": published.isoformat() + "T00:00:00Z",
            "query_source": f"{keyword}|{published:%Y-%m}",
            "search_window": f"{published:%Y-%m}",
            "view_count": int(5000 + idx * 750 + rng.randint(0, 3000)),
            "like_count": int(100 + idx * 20 + rng.randint(0, 200)),
            "comment_count": int(12 + idx * 4 + rng.randint(0, 50)),
            "retrieved_at": utc_now_iso(),
            "relevance_status": "pending",
            "source": "demo",
        }
        rows.append(row)
    return rows


def _merge_query_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        video_id = record["video_id"]
        current = merged.setdefault(video_id, dict(record))
        if current is not record:
            current["view_count"] = max(int(current.get("view_count") or 0), int(record.get("view_count") or 0))
            current["like_count"] = max(int(current.get("like_count") or 0), int(record.get("like_count") or 0))
            current["comment_count"] = max(int(current.get("comment_count") or 0), int(record.get("comment_count") or 0))
            sources = set()
            for value in [current.get("query_source"), record.get("query_source")]:
                if isinstance(value, list):
                    sources.update(map(str, value))
                elif value:
                    sources.add(str(value))
            current["query_source"] = sorted(sources)
    return list(merged.values())


def _search_one_query(
    ctx,
    logger,
    keyword: str,
    window_start: dt.date,
    window_end: dt.date,
    max_pages: int,
) -> list[dict[str, Any]]:
    api_key = ctx.youtube_api_key
    if not api_key:
        raise RuntimeError("YouTube API key is not configured.")
    published_after = iso_datetime_for_date(window_start)
    published_before = iso_datetime_for_date(window_end, end_of_day=True)
    page_token: str | None = None
    collected: list[dict[str, Any]] = []
    for page_index in range(max_pages):
        params = {
            "key": api_key,
            "part": "snippet",
            "type": "video",
            "q": keyword,
            "regionCode": ctx.keywords_config.get("youtube_search", {}).get("region_code", "AU"),
            "relevanceLanguage": ctx.keywords_config.get("youtube_search", {}).get("relevance_language", "en"),
            "maxResults": int(ctx.keywords_config.get("youtube_search", {}).get("max_results_per_page", 50)),
            "publishedAfter": published_after,
            "publishedBefore": published_before,
        }
        if page_token:
            params["pageToken"] = page_token
        payload = request_json_with_backoff(SEARCH_URL, params=params, logger=logger)
        append_jsonl(
            ctx.paths.raw / "youtube_search_responses.jsonl",
            {
                "retrieved_at": utc_now_iso(),
                "keyword": keyword,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "page_index": page_index,
                "page_token": page_token,
                "source": "youtube_search",
                "payload": payload,
            },
        )
        items = payload.get("items", [])
        video_ids = [item.get("id", {}).get("videoId") for item in items if item.get("id", {}).get("videoId")]
        details = fetch_video_statistics(ctx, logger, video_ids)
        detail_by_id = {item["video_id"]: item for item in details}
        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            stats = detail_by_id.get(video_id, {})
            collected.append(
                {
                    "video_id": video_id,
                    "title": normalize_text(snippet.get("title")),
                    "description": normalize_text(snippet.get("description")),
                    "channel_id": snippet.get("channelId"),
                    "channel_name": normalize_text(snippet.get("channelTitle")),
                    "published_at": snippet.get("publishedAt"),
                    "query_source": f"{keyword}|{window_start:%Y-%m}",
                    "search_window": f"{window_start:%Y-%m}",
                    "view_count": stats.get("view_count"),
                    "like_count": stats.get("like_count"),
                    "comment_count": stats.get("comment_count"),
                    "retrieved_at": utc_now_iso(),
                    "relevance_status": "pending",
                    "source": "youtube_search",
                }
            )
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return collected


def fetch_video_statistics(ctx, logger, video_ids: list[str]) -> list[dict[str, Any]]:
    if not video_ids or not ctx.youtube_api_key:
        return []
    rows: list[dict[str, Any]] = []
    for batch in chunked(video_ids, 50):
        params = {
            "key": ctx.youtube_api_key,
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "maxResults": 50,
        }
        payload = request_json_with_backoff(VIDEOS_URL, params=params, logger=logger)
        append_jsonl(
            ctx.paths.raw / "youtube_video_details.jsonl",
            {
                "retrieved_at": utc_now_iso(),
                "source": "youtube_video_details",
                "payload": payload,
            },
        )
        for item in payload.get("items", []):
            statistics = item.get("statistics", {})
            rows.append(
                {
                    "video_id": item.get("id"),
                    "view_count": int(statistics.get("viewCount") or 0),
                    "like_count": int(statistics.get("likeCount") or 0),
                    "comment_count": int(statistics.get("commentCount") or 0),
                }
            )
    return rows


def run(
    ctx=None,
    demo: bool = False,
    limit: int | None = None,
    refresh: bool = False,
    demo_scale: int = 1,
    max_jobs: int | None = None,
) -> pd.DataFrame:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("youtube_search")
    logger.info("starting search demo=%s", demo)
    flat_path = ctx.paths.interim / "videos_search_flat.jsonl"
    if flat_path.exists() and flat_path.stat().st_size > 0 and not demo and not refresh:
        logger.info("search output already exists; loading %s", flat_path)
        return pd.DataFrame(pd.read_json(flat_path, lines=True))
    if demo or not ctx.youtube_api_key:
        rows = _demo_video_rows(ctx, scale=demo_scale)
        if limit:
            rows = rows[:limit]
        raw_envelopes = [
            {
                "retrieved_at": utc_now_iso(),
                "source": "demo",
                "query": row["query_source"],
                "payload": row,
            }
            for row in rows
        ]
        write_jsonl(ctx.paths.raw / "youtube_search_responses.jsonl", raw_envelopes)
        write_jsonl(flat_path, rows)
        logger.info("wrote %s demo videos", len(rows))
        return pd.DataFrame(rows)
    jobs = build_query_jobs(ctx)
    if max_jobs:
        jobs = jobs[:max_jobs]
    max_pages = int(ctx.keywords_config.get("youtube_search", {}).get("max_pages_per_query_window", 5))
    max_pages = int(os.getenv("YOUTUBE_MAX_PAGES_PER_QUERY_WINDOW", max_pages))
    rows: list[dict[str, Any]] = []
    for job in jobs:
        logger.info(
            "query keyword=%s window=%s",
            job["keyword"],
            job["window_label"],
        )
        rows.extend(
            _search_one_query(
                ctx=ctx,
                logger=logger,
                keyword=job["keyword"],
                window_start=job["window_start"],
                window_end=job["window_end"],
                max_pages=max_pages,
            )
        )
        interim_rows = _merge_query_sources(rows)
        if limit:
            interim_rows = interim_rows[:limit]
        write_jsonl(flat_path, interim_rows)
    rows = _merge_query_sources(rows)
    if limit:
        rows = rows[:limit]
    write_jsonl(flat_path, rows)
    logger.info("wrote %s search records", len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect YouTube search results.")
    parser.add_argument("--demo", action="store_true", help="Write deterministic demo search data instead of calling the API.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows written.")
    parser.add_argument("--refresh", action="store_true", help="Regenerate and overwrite existing search output.")
    parser.add_argument("--demo-scale", type=int, default=1, help="Multiply the demo video pool when running without an API key.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit keyword/month search jobs for quota-safe tests.")
    args = parser.parse_args()
    run(demo=args.demo, limit=args.limit, refresh=args.refresh, demo_scale=args.demo_scale, max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
