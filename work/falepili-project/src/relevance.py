from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .common import ensure_dirs, load_context, load_table, normalize_text, save_table, setup_logging


RELEVANCE_TERMS = [
    "falepili",
    "tuvalu",
    "mobility pathway",
    "migration",
    "climate mobility",
    "climate refugee",
    "security",
    "treaty",
    "australia-tuvalu",
]


def _score_text(text: str) -> int:
    text_l = normalize_text(text).lower()
    score = sum(1 for term in RELEVANCE_TERMS if term in text_l)
    if "australia" in text_l and "tuvalu" in text_l:
        score += 2
    return score


def label_relevance(text: str) -> str:
    score = _score_text(text)
    if score >= 4:
        return "directly relevant"
    if score >= 2:
        return "contextually relevant"
    if len(normalize_text(text)) < 20:
        return "unclear"
    return "irrelevant"


def score_videos(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos
    out = videos.copy()
    combined = out["title"].fillna("") + " " + out["description"].fillna("")
    out["relevance_status"] = combined.map(label_relevance)
    out["relevance_score"] = combined.map(_score_text)
    return out


def score_comments(comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return comments
    out = comments.copy()
    video_lookup = videos.set_index("video_id")["title"].to_dict() if not videos.empty else {}
    combined = out["text"].fillna("")
    if "video_id" in out.columns:
        combined = combined + " " + out["video_id"].map(video_lookup).fillna("")
    out["relevance_status"] = combined.map(label_relevance)
    out["relevance_score"] = combined.map(_score_text)
    return out


def run(ctx=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("relevance")
    videos = load_table(ctx.paths.processed / "videos.parquet") if (ctx.paths.processed / "videos.parquet").exists() else pd.DataFrame()
    comments = load_table(ctx.paths.processed / "comments.parquet") if (ctx.paths.processed / "comments.parquet").exists() else pd.DataFrame()
    videos_scored = score_videos(videos)
    comments_scored = score_comments(comments, videos_scored)
    if not videos_scored.empty:
        save_table(videos_scored, ctx.paths.processed / "videos_relevance.parquet")
    if not comments_scored.empty:
        save_table(comments_scored, ctx.paths.processed / "comments_relevance.parquet")
    logger.info("scored videos=%s comments=%s", len(videos_scored), len(comments_scored))
    return videos_scored, comments_scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign heuristic relevance labels.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

