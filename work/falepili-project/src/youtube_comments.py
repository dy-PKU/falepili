from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    append_jsonl,
    chunked,
    ensure_dirs,
    load_context,
    normalize_text,
    read_jsonl,
    request_json_with_backoff,
    setup_logging,
    stable_hash,
    utc_now_iso,
    write_jsonl,
)


COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/comments"


TOP_LEVEL_TEMPLATES = [
    ("This is a smart regional arrangement. People need to understand the climate and security context.", 11),
    ("I worry about housing, services and whether this is being sold as a simple solution.", 4),
    ("Tuvalu should not be left alone. Australia has a responsibility here.", 15),
    ("Feels like the media keeps framing this through geopolitics instead of human security.", 8),
    ("What happens when the pathway is actually implemented? This needs details.", 6),
    ("This is just fear-mongering and cynical politics.", 3),
]

REPLY_TEMPLATES = [
    ("Exactly, the policy is more complex than a headline.", 5),
    ("I disagree. The costs and sovereignty issues matter too.", 2),
    ("That is a fair point about Pacific solidarity.", 7),
    ("Not every migration issue is a crisis story.", 4),
    ("The humanitarian angle should be front and centre.", 9),
]


def _load_video_index(ctx, videos_path: Path | None = None) -> list[dict[str, Any]]:
    videos_path = videos_path or (ctx.paths.interim / "videos_search_flat.jsonl")
    if videos_path.exists():
        return read_jsonl(videos_path)
    processed = ctx.paths.processed / "videos.parquet"
    if processed.exists():
        return pd.read_parquet(processed).to_dict(orient="records")
    return []


def _demo_comments(ctx, videos: list[dict[str, Any]], scale: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(ctx.random_seed + 7)
    comments: list[dict[str, Any]] = []
    scaled_videos = videos * max(1, int(scale))
    for video_idx, video in enumerate(scaled_videos):
        video_id = video["video_id"]
        published_base = dt.datetime.fromisoformat(str(video["published_at"]).replace("Z", "+00:00"))
        for thread_idx in range(3):
            text, base_likes = TOP_LEVEL_TEMPLATES[(video_idx + thread_idx) % len(TOP_LEVEL_TEMPLATES)]
            comment_id = f"demo_c_{stable_hash([video_id, thread_idx, 'top'])[:12]}"
            thread_id = f"thread_{video_id}_{thread_idx}"
            reply_total = rng.randint(0, 4)
            top_comment = {
                "comment_id": comment_id,
                "video_id": video_id,
                "parent_id": None,
                "thread_id": thread_id,
                "text": text,
                "like_count": int(base_likes + rng.randint(0, 25)),
                "published_at": (published_base + dt.timedelta(days=thread_idx + rng.randint(0, 20))).isoformat().replace("+00:00", "Z"),
                "updated_at": (published_base + dt.timedelta(days=thread_idx + rng.randint(0, 20), hours=2)).isoformat().replace("+00:00", "Z"),
                "is_reply": False,
                "retrieved_at": utc_now_iso(),
                "source": "demo",
            }
            comments.append(top_comment)
            for reply_idx in range(reply_total):
                reply_text, reply_base = REPLY_TEMPLATES[(video_idx + thread_idx + reply_idx) % len(REPLY_TEMPLATES)]
                reply_comment = {
                    "comment_id": f"demo_r_{stable_hash([video_id, thread_idx, reply_idx])[:12]}",
                    "video_id": video_id,
                    "parent_id": comment_id,
                    "thread_id": thread_id,
                    "text": reply_text,
                    "like_count": int(reply_base + rng.randint(0, 12)),
                    "published_at": (published_base + dt.timedelta(days=thread_idx + reply_idx + rng.randint(0, 25))).isoformat().replace("+00:00", "Z"),
                    "updated_at": (published_base + dt.timedelta(days=thread_idx + reply_idx + rng.randint(0, 25), hours=1)).isoformat().replace("+00:00", "Z"),
                    "is_reply": True,
                    "retrieved_at": utc_now_iso(),
                    "source": "demo",
                }
                comments.append(reply_comment)
    return comments


def _parse_thread_payload(video_id: str, thread: dict[str, Any], retrieved_at: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    thread_id = thread.get("id")
    top = thread.get("snippet", {}).get("topLevelComment", {})
    top_snippet = top.get("snippet", {})
    comment_id = top.get("id")
    if comment_id:
        results.append(
            {
                "comment_id": comment_id,
                "video_id": video_id,
                "parent_id": None,
                "thread_id": thread_id,
                "text": normalize_text(top_snippet.get("textOriginal") or top_snippet.get("textDisplay")),
                "like_count": int(top_snippet.get("likeCount") or 0),
                "published_at": top_snippet.get("publishedAt"),
                "updated_at": top_snippet.get("updatedAt"),
                "is_reply": False,
                "retrieved_at": retrieved_at,
                "source": "youtube_commentThreads",
            }
        )
    for reply in thread.get("replies", {}).get("comments", []) or []:
        r_snippet = reply.get("snippet", {})
        results.append(
            {
                "comment_id": reply.get("id"),
                "video_id": video_id,
                "parent_id": comment_id,
                "thread_id": thread_id,
                "text": normalize_text(r_snippet.get("textOriginal") or r_snippet.get("textDisplay")),
                "like_count": int(r_snippet.get("likeCount") or 0),
                "published_at": r_snippet.get("publishedAt"),
                "updated_at": r_snippet.get("updatedAt"),
                "is_reply": True,
                "retrieved_at": retrieved_at,
                "source": "youtube_commentThreads",
            }
        )
    return results


def _fetch_replies(
    ctx,
    logger,
    parent_id: str,
    video_id: str,
    thread_id: str,
    max_replies: int | None = None,
) -> list[dict[str, Any]]:
    if not ctx.youtube_api_key:
        return []
    collected: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "key": ctx.youtube_api_key,
            "part": "snippet",
            "parentId": parent_id,
            "textFormat": "plainText",
            "maxResults": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            payload = request_json_with_backoff(COMMENTS_URL, params=params, logger=logger)
        except RuntimeError as exc:
            logger.warning("reply_fetch_failed video_id=%s parent_id=%s", video_id, parent_id)
            break
        append_jsonl(
            ctx.paths.raw / "youtube_comment_replies.jsonl",
            {
                "retrieved_at": utc_now_iso(),
                "video_id": video_id,
                "parent_id": parent_id,
                "thread_id": thread_id,
                "payload": payload,
            },
        )
        for item in payload.get("items", []):
            if max_replies is not None and len(collected) >= max_replies:
                break
            snippet = item.get("snippet", {})
            collected.append(
                {
                    "comment_id": item.get("id"),
                    "video_id": video_id,
                    "parent_id": parent_id,
                    "thread_id": thread_id,
                    "text": normalize_text(snippet.get("textOriginal") or snippet.get("textDisplay")),
                    "like_count": int(snippet.get("likeCount") or 0),
                    "published_at": snippet.get("publishedAt"),
                    "updated_at": snippet.get("updatedAt"),
                    "is_reply": True,
                    "retrieved_at": utc_now_iso(),
                    "source": "youtube_comments",
                }
            )
        page_token = payload.get("nextPageToken")
        if not page_token or (max_replies is not None and len(collected) >= max_replies):
            break
    return collected


def _fetch_comments_for_video(
    ctx,
    logger,
    video_id: str,
    max_threads: int | None = None,
    max_replies_per_thread: int | None = None,
) -> list[dict[str, Any]]:
    if not ctx.youtube_api_key:
        return []
    collected: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "key": ctx.youtube_api_key,
            "part": "snippet,replies",
            "videoId": video_id,
            "textFormat": "plainText",
            "maxResults": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            payload = request_json_with_backoff(COMMENT_THREADS_URL, params=params, logger=logger)
        except RuntimeError as exc:
            logger.warning("comment_thread_failed video_id=%s", video_id)
            break
        retrieved_at = utc_now_iso()
        append_jsonl(
            ctx.paths.raw / "youtube_comment_threads.jsonl",
            {
                "retrieved_at": retrieved_at,
                "video_id": video_id,
                "payload": payload,
            },
        )
        for thread in payload.get("items", []):
            top_level_count = sum(1 for row in collected if not row.get("is_reply"))
            if max_threads is not None and top_level_count >= max_threads:
                break
            parsed = _parse_thread_payload(video_id, thread, retrieved_at)
            collected.extend(parsed)
            if parsed:
                top_level = parsed[0]
                collected.extend(
                    _fetch_replies(
                        ctx,
                        logger,
                        top_level["comment_id"],
                        video_id,
                        top_level["thread_id"],
                        max_replies=max_replies_per_thread,
                    )
                )
        page_token = payload.get("nextPageToken")
        top_level_count = sum(1 for row in collected if not row.get("is_reply"))
        if not page_token or (max_threads is not None and top_level_count >= max_threads):
            break
    return collected


def run(
    ctx=None,
    demo: bool = False,
    limit: int | None = None,
    videos_path: Path | None = None,
    refresh: bool = False,
    append: bool = False,
    demo_scale: int = 1,
    max_videos: int | None = None,
    max_comments: int | None = None,
    max_threads_per_video: int | None = None,
    max_replies_per_thread: int | None = None,
) -> pd.DataFrame:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("youtube_comments")
    logger.info("starting comments demo=%s", demo)
    flat_path = ctx.paths.interim / "comments_flat.jsonl"
    if flat_path.exists() and flat_path.stat().st_size > 0 and not demo and not refresh and not append:
        logger.info("comments output already exists; loading %s", flat_path)
        return pd.DataFrame(read_jsonl(flat_path))
    existing_rows = read_jsonl(flat_path) if append and flat_path.exists() else []
    videos = _load_video_index(ctx, videos_path=videos_path)
    video_limit = max_videos or limit
    if video_limit:
        videos = videos[:video_limit]
    if not videos:
        logger.warning("no videos available for comment collection")
        write_jsonl(flat_path, [])
        return pd.DataFrame()
    if demo or not ctx.youtube_api_key:
        rows = _demo_comments(ctx, videos, scale=demo_scale)
        if limit:
            rows = rows[: limit * 6]
        write_jsonl(ctx.paths.raw / "youtube_comment_threads.jsonl", [{"source": "demo", "payload": "synthetic"}])
        if append and existing_rows:
            rows = _merge_comments(existing_rows, rows)
        write_jsonl(flat_path, rows)
        logger.info("wrote %s demo comments", len(rows))
        return pd.DataFrame(rows)
    rows: list[dict[str, Any]] = []
    for video in videos:
        rows.extend(
            _fetch_comments_for_video(
                ctx,
                logger,
                video["video_id"],
                max_threads=max_threads_per_video,
                max_replies_per_thread=max_replies_per_thread,
            )
        )
        if max_comments is not None and len(rows) >= max_comments:
            rows = rows[:max_comments]
            break
    if max_comments is not None:
        rows = rows[:max_comments]
    if append and existing_rows:
        rows = _merge_comments(existing_rows, rows)
    write_jsonl(flat_path, rows)
    logger.info("wrote %s comments", len(rows))
    return pd.DataFrame(rows)


def _merge_comments(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows + new_rows:
        comment_id = row.get("comment_id")
        if comment_id:
            merged[str(comment_id)] = row
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect YouTube comments.")
    parser.add_argument("--demo", action="store_true", help="Write deterministic demo comments instead of calling the API.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos or comments processed.")
    parser.add_argument("--videos-path", type=str, default=None, help="Optional path to a videos search JSONL or parquet file.")
    parser.add_argument("--refresh", action="store_true", help="Regenerate and overwrite existing comment output.")
    parser.add_argument("--append", action="store_true", help="Fetch selected videos and merge them into existing comment output.")
    parser.add_argument("--demo-scale", type=int, default=1, help="Multiply the demo comment pool when running without an API key.")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit videos processed for comment collection.")
    parser.add_argument("--max-comments", type=int, default=None, help="Stop after this many collected comments.")
    parser.add_argument("--max-threads-per-video", type=int, default=None, help="Limit top-level comment threads per video.")
    parser.add_argument("--max-replies-per-thread", type=int, default=None, help="Limit replies fetched per top-level comment.")
    args = parser.parse_args()
    run(
        demo=args.demo,
        limit=args.limit,
        videos_path=Path(args.videos_path) if args.videos_path else None,
        refresh=args.refresh,
        append=args.append,
        demo_scale=args.demo_scale,
        max_videos=args.max_videos,
        max_comments=args.max_comments,
        max_threads_per_video=args.max_threads_per_video,
        max_replies_per_thread=args.max_replies_per_thread,
    )


if __name__ == "__main__":
    main()
