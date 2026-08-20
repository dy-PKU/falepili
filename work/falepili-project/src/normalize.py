from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .common import ensure_dirs, load_context, normalize_text, read_jsonl, save_table, setup_logging


def _merge_query_sources(values: pd.Series) -> str:
    sources: set[str] = set()
    for value in values.dropna().tolist():
        if isinstance(value, list):
            sources.update(str(item) for item in value if str(item).strip())
        else:
            text = str(value).strip()
            if text:
                for part in text.split(";"):
                    part = part.strip()
                    if part:
                        sources.add(part)
    return "; ".join(sorted(sources))


def _read_jsonl_or_empty(path: Path) -> pd.DataFrame:
    records = read_jsonl(path)
    return pd.DataFrame(records)


def normalize_videos(ctx) -> pd.DataFrame:
    videos_path = ctx.paths.interim / "videos_search_flat.jsonl"
    if videos_path.exists():
        df = _read_jsonl_or_empty(videos_path)
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    df["title"] = df["title"].map(normalize_text)
    df["description"] = df["description"].map(normalize_text)
    df["channel_name"] = df["channel_name"].map(normalize_text)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
    df["view_count"] = pd.to_numeric(df.get("view_count"), errors="coerce").fillna(0).astype(int)
    df["like_count"] = pd.to_numeric(df.get("like_count"), errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df.get("comment_count"), errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["video_id", "retrieved_at"]).groupby("video_id", as_index=False).agg(
        {
            "title": "last",
            "description": "last",
            "channel_id": "last",
            "channel_name": "last",
            "published_at": "min",
            "query_source": _merge_query_sources,
            "search_window": "last",
            "view_count": "max",
            "like_count": "max",
            "comment_count": "max",
            "retrieved_at": "max",
            "relevance_status": "last",
            "source": "last",
        }
    )
    df["published_month"] = df["published_at"].dt.tz_convert(None).dt.to_period("M").astype(str)
    df["video_age_days"] = (pd.Timestamp.now(tz="UTC") - df["published_at"]).dt.days
    save_table(df, ctx.paths.processed / "videos.parquet")
    return df


def normalize_comments(ctx) -> pd.DataFrame:
    comments_path = ctx.paths.interim / "comments_flat.jsonl"
    if comments_path.exists():
        df = _read_jsonl_or_empty(comments_path)
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    df["text"] = df["text"].map(normalize_text)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
    df["like_count"] = pd.to_numeric(df.get("like_count"), errors="coerce").fillna(0).astype(int)
    df["is_reply"] = df.get("is_reply", False).fillna(False).astype(bool)
    df = df.drop_duplicates(subset=["comment_id"], keep="last")
    df["text_original"] = df["text"]
    df["text_clean"] = df["text"].map(normalize_text)
    df["comment_age_days"] = (pd.Timestamp.now(tz="UTC") - df["published_at"]).dt.days
    save_table(df, ctx.paths.processed / "comments.parquet")
    return df


def run(ctx=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("normalize")
    videos = normalize_videos(ctx)
    comments = normalize_comments(ctx)
    logger.info("normalized videos=%s comments=%s", len(videos), len(comments))
    return videos, comments


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw YouTube data into parquet tables.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
