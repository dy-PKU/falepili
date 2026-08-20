from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .common import (
    ensure_dirs,
    chunked,
    load_context,
    load_table,
    normalize_text,
    save_table,
    setup_logging,
    stable_hash,
    utc_now_iso,
    write_json,
)


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _openai_endpoint() -> str:
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_OPENAI_BASE_URL
    ).rstrip("/")
    if base_url.endswith("/responses"):
        return base_url
    return f"{base_url}/responses"


def _schema(ctx) -> dict[str, Any]:
    cfg = ctx.keywords_config.get("coding", {})
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relevance": {"type": "string", "enum": cfg.get("relevance_labels", [])},
            "stance": {"type": "string", "enum": cfg.get("stances", [])},
            "frames": {
                "type": "array",
                "items": {"type": "string", "enum": cfg.get("frames", [])},
            },
            "emotion": {"type": "string", "enum": cfg.get("emotions", [])},
            "narrative_frame": {"type": "string", "enum": cfg.get("narrative_frames", [])},
            "language_features": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    feat: {"type": "boolean"} for feat in cfg.get("language_features", [])
                },
                "required": cfg.get("language_features", []),
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "relevance",
            "stance",
            "frames",
            "emotion",
            "narrative_frame",
            "language_features",
            "rationale",
        ],
    }


def _prompt_text(row: pd.Series, kind: str) -> str:
    if kind == "video":
        return f"Title: {row.get('title', '')}\nDescription: {row.get('description', '')}"
    if kind == "reddit_post":
        return (
            f"Platform: Reddit\nSubreddit: r/{row.get('subreddit', '')}\n"
            f"Post title: {row.get('title', '')}\nPost text: {row.get('selftext', '')}"
        )
    if kind == "reddit_comment":
        return (
            f"Platform: Reddit\nSubreddit: r/{row.get('subreddit', '')}\n"
            f"Thread title: {row.get('thread_title', '')}\n"
            f"Comment: {row.get('body', '')}"
        )
    return f"Comment: {row.get('text', '')}"


def _heuristic_code(text: str, ctx) -> dict[str, Any]:
    text_l = normalize_text(text).lower()
    coding = ctx.keywords_config.get("coding", {})
    relevance = "directly relevant" if any(term in text_l for term in ["falepili", "tuvalu", "mobility", "security"]) else "contextually relevant" if "australia" in text_l else "irrelevant"
    if "?" in text_l:
        stance = "Concern"
    elif any(term in text_l for term in ["support", "good", "smart", "needed", "responsibility", "solidarity"]):
        stance = "Support"
    elif any(term in text_l for term in ["fear", "cost", "worry", "problem", "crisis"]):
        stance = "Concern"
    elif any(term in text_l for term in ["bad", "oppose", "wrong", "cynical", "not"]):
        stance = "Strong Opposition"
    else:
        stance = "Neutral"
    if "climate" in text_l or "warming" in text_l:
        frames = ["Climate Justice", "Human Security"]
    elif "security" in text_l or "geopolitics" in text_l:
        frames = ["Strategic Security", "China / Geopolitics"]
    elif "housing" in text_l or "services" in text_l:
        frames = ["Housing / Public Services"]
    elif "migration" in text_l or "visa" in text_l:
        frames = ["Migration Fairness"]
    else:
        frames = ["Other"]
    if any(term in text_l for term in ["sorry", "sad", "empathy"]):
        emotion = "Empathy"
    elif any(term in text_l for term in ["support", "solidarity"]):
        emotion = "Solidarity"
    elif any(term in text_l for term in ["hope", "needed"]):
        emotion = "Hope"
    elif any(term in text_l for term in ["worry", "fear", "anxious"]):
        emotion = "Anxiety"
    elif any(term in text_l for term in ["angry", "mad", "outrage"]):
        emotion = "Anger"
    elif any(term in text_l for term in ["cynical", "sceptical", "skeptical"]):
        emotion = "Cynicism"
    else:
        emotion = "Neutral"
    if "refugee" in text_l or "displaced" in text_l:
        narrative = "Refugees"
    elif "worker" in text_l:
        narrative = "Workers"
    elif "citizen" in text_l or "pathway" in text_l:
        narrative = "New citizens"
    elif "partner" in text_l or "alliance" in text_l:
        narrative = "Strategic partners"
    elif "welfare" in text_l or "benefit" in text_l:
        narrative = "Welfare recipients"
    elif "pacific" in text_l:
        narrative = "Pacific neighbours"
    else:
        narrative = "Other"
    return {
        "relevance": relevance,
        "stance": stance,
        "frames": frames,
        "emotion": emotion,
        "narrative_frame": narrative,
        "language_features": {
            "sarcasm": "yeah right" in text_l or "sure" in text_l,
            "irony": "iron" in text_l,
            "rhetorical_question": "?" in text_l,
            "metaphor": any(term in text_l for term in ["like a", "as if"]),
        },
        "rationale": "Heuristic fallback used because no OpenAI API key was available.",
    }


def _build_openai_payload(prompt: str, ctx) -> dict[str, Any]:
    return {
        "model": ctx.openai_model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are coding Reddit and YouTube public discourse about Australia, Tuvalu, "
                            "the Falepili Union, climate mobility, migration, and security cooperation. "
                            "Interpret comments in their thread context, distinguish the author's stance "
                            "from quoted material, and do not infer demographic or geographic attributes. "
                            "Return only valid JSON that matches the provided schema."
                        ),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "falepili_comment_code",
                "schema": _schema(ctx),
                "strict": True,
            }
        },
    }


def _call_openai(prompt: str, ctx) -> dict[str, Any]:
    if not ctx.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    payload = _build_openai_payload(prompt, ctx)
    response = requests.post(
        _openai_endpoint(),
        headers={"Authorization": f"Bearer {ctx.openai_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("output_text")
    if not text:
        for block in data.get("output", []):
            for content in block.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text")
                    break
            if text:
                break
    if not text:
        raise RuntimeError("OpenAI response did not include output_text")
    return json.loads(text)


def _openai_error_payload(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    body = None
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = response.text[:1000] if getattr(response, "text", None) else None
    return {
        "error_type": exc.__class__.__name__,
        "status_code": status_code,
        "body": body,
    }


def _code_frame_row(
    row: pd.Series,
    ctx,
    kind: str,
    cache_dir: Path,
    allow_heuristic: bool = True,
) -> dict[str, Any]:
    prompt = _prompt_text(row, kind)
    force_heuristic = os.getenv("LLM_FORCE_HEURISTIC", "").strip().lower() in {"1", "true", "yes", "on"}
    coding_backend = "openai" if ctx.openai_api_key and not force_heuristic else "heuristic"
    cache_key = stable_hash(
        {"kind": kind, "prompt": prompt, "model": ctx.openai_model, "backend": coding_backend}
    )
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if not ctx.openai_api_key or cached.get("source") == "openai":
            return cached
    openai_error = None
    try:
        if force_heuristic:
            raise RuntimeError("LLM_FORCE_HEURISTIC is enabled")
        coded = _call_openai(prompt, ctx)
        coded["source"] = "openai"
    except Exception as exc:
        if not allow_heuristic:
            raise RuntimeError(f"OpenAI coding failed for kind={kind}: {_openai_error_payload(exc)}") from exc
        openai_error = _openai_error_payload(exc)
        coded = _heuristic_code(prompt, ctx)
        coded["source"] = "heuristic"
    coded["openai_error"] = openai_error
    coded["cached_at"] = utc_now_iso()
    coded["cache_key"] = cache_key
    write_json(cache_path, coded)
    return coded


def _code_table(
    df: pd.DataFrame,
    ctx,
    kind: str,
    allow_heuristic: bool = True,
    batch_size: int | None = None,
    output_path: Path | None = None,
    logger=None,
) -> pd.DataFrame:
    if df.empty:
        return df
    cache_dir = ctx.paths.llm_cache / kind
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    batch_size = max(1, int(batch_size or ctx.llm_batch_size or 25))
    total_batches = max(1, (len(df) + batch_size - 1) // batch_size)
    for batch_index, batch_rows in enumerate(chunked(list(df.iterrows()), batch_size), start=1):
        if logger is not None:
            logger.info("coding_%s_batch batch=%s/%s rows=%s", kind, batch_index, total_batches, len(batch_rows))
        for _, row in batch_rows:
            coded = _code_frame_row(
                row,
                ctx,
                kind=kind,
                cache_dir=cache_dir,
                allow_heuristic=allow_heuristic,
            )
            record = row.to_dict()
            record.update(
                {
                    "llm_relevance": coded.get("relevance"),
                    "stance": coded.get("stance"),
                    "frames": coded.get("frames", []),
                    "emotion": coded.get("emotion"),
                    "narrative_frame": coded.get("narrative_frame"),
                    "sarcasm": bool(coded.get("language_features", {}).get("sarcasm")),
                    "irony": bool(coded.get("language_features", {}).get("irony")),
                    "rhetorical_question": bool(coded.get("language_features", {}).get("rhetorical_question")),
                    "metaphor": bool(coded.get("language_features", {}).get("metaphor")),
                    "llm_rationale": coded.get("rationale"),
                    "llm_source": coded.get("source"),
                    "llm_openai_error": coded.get("openai_error"),
                    "llm_cached_at": coded.get("cached_at"),
                    "llm_cache_key": coded.get("cache_key"),
                }
            )
            records.append(record)
        if output_path is not None:
            save_table(pd.DataFrame(records), output_path)
    return pd.DataFrame(records)


def run(ctx=None, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("llm_coding")
    videos = load_table(ctx.paths.processed / "videos_relevance.parquet") if (ctx.paths.processed / "videos_relevance.parquet").exists() else load_table(ctx.paths.processed / "videos.parquet")
    comments = load_table(ctx.paths.processed / "comments_relevance.parquet") if (ctx.paths.processed / "comments_relevance.parquet").exists() else load_table(ctx.paths.processed / "comments.parquet")
    if limit:
        comments = comments.head(limit).copy()
        videos = videos.head(limit).copy()
    videos_coded = _code_table(
        videos,
        ctx,
        kind="video",
        batch_size=ctx.llm_batch_size,
        output_path=ctx.paths.processed / "videos_coded.parquet",
        logger=logger,
    )
    comments_coded = _code_table(
        comments,
        ctx,
        kind="comment",
        batch_size=ctx.llm_batch_size,
        output_path=ctx.paths.processed / "comments_coded.parquet",
        logger=logger,
    )
    logger.info("coded videos=%s comments=%s", len(videos_coded), len(comments_coded))
    return videos_coded, comments_coded


def run_reddit(
    ctx=None,
    limit: int | None = None,
    allow_heuristic: bool = False,
    comments_only: bool = False,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Code normalized Reddit posts and comments with thread context."""
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("llm_coding_reddit")
    if not ctx.openai_api_key and not allow_heuristic:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Formal Reddit coding refuses heuristic fallback; "
            "configure .env or pass --allow-heuristic for a structure-only smoke test."
        )

    posts_path = ctx.paths.processed / "reddit_posts.parquet"
    comments_path = ctx.paths.processed / "reddit_comments.parquet"
    posts = load_table(posts_path) if posts_path.exists() else pd.DataFrame()
    comments = load_table(comments_path) if comments_path.exists() else pd.DataFrame()
    if not comments.empty:
        comments = comments[comments.get("content_status", "available") == "available"].copy()
        context = posts[["reddit_post_id", "title", "subreddit"]].rename(
            columns={"title": "thread_title", "subreddit": "thread_subreddit"}
        )
        comments = comments.merge(context, on="reddit_post_id", how="left")
        if "subreddit" not in comments.columns:
            comments["subreddit"] = comments["thread_subreddit"]
        else:
            comments["subreddit"] = comments["subreddit"].fillna(comments["thread_subreddit"])
        comments = comments.drop(columns=["thread_subreddit"], errors="ignore")
    if limit:
        posts = posts.head(limit).copy()
        comments = comments.head(limit).copy()

    posts_coded = (
        pd.DataFrame()
        if comments_only
        else _code_table(
            posts,
            ctx,
            kind="reddit_post",
            allow_heuristic=allow_heuristic,
            batch_size=batch_size,
            output_path=ctx.paths.processed / "reddit_posts_coded.parquet",
            logger=logger,
        )
    )
    comments_coded = _code_table(
        comments,
        ctx,
        kind="reddit_comment",
        allow_heuristic=allow_heuristic,
        batch_size=batch_size,
        output_path=ctx.paths.processed / "reddit_comments_coded.parquet",
        logger=logger,
    )
    coded_tables = [df for df in [posts_coded, comments_coded] if not df.empty and "llm_source" in df.columns]
    used_heuristic = any((df["llm_source"] == "heuristic").any() for df in coded_tables)
    suffix = "_coded_heuristic.parquet" if used_heuristic else "_coded.parquet"
    if not posts_coded.empty:
        save_table(posts_coded, ctx.paths.processed / f"reddit_posts{suffix}")
    if not comments_coded.empty:
        save_table(comments_coded, ctx.paths.processed / f"reddit_comments{suffix}")
    logger.info("coded reddit_posts=%s reddit_comments=%s", len(posts_coded), len(comments_coded))
    return posts_coded, comments_coded


def main() -> None:
    parser = argparse.ArgumentParser(description="Code videos and comments with an LLM or heuristic fallback.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--source", choices=["youtube", "reddit"], default="youtube")
    parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Allow heuristic fallback for structure-only smoke tests; do not use as research coding.",
    )
    parser.add_argument("--comments-only", action="store_true", help="Code Reddit comments but not posts.")
    parser.add_argument("--batch-size", type=int, default=None, help="Rows per LLM batch; defaults to LLM_BATCH_SIZE.")
    args = parser.parse_args()
    if args.source == "reddit":
        run_reddit(
            limit=args.limit,
            allow_heuristic=args.allow_heuristic,
            comments_only=args.comments_only,
            batch_size=args.batch_size,
        )
    else:
        run(limit=args.limit)


if __name__ == "__main__":
    main()
