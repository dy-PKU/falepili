from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import ensure_dirs, js_divergence, listify_cell, load_context, load_table, save_table, setup_logging


def _load_tables(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_path = ctx.paths.processed / "videos_coded.parquet"
    comments_path = ctx.paths.processed / "comments_coded.parquet"
    if not videos_path.exists():
        videos_path = ctx.paths.processed / "videos_relevance.parquet"
    if not comments_path.exists():
        comments_path = ctx.paths.processed / "comments_relevance.parquet"
    videos = load_table(videos_path) if videos_path.exists() else pd.DataFrame()
    comments = load_table(comments_path) if comments_path.exists() else pd.DataFrame()
    return videos, comments


def _explode_frames(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty or "frames" not in df.columns:
        return pd.DataFrame(columns=[id_col, "frame"])
    out = df[[id_col, "frames"]].copy()
    out["frames"] = out["frames"].apply(listify_cell)
    return out.explode("frames").rename(columns={"frames": "frame"})


def _frame_distribution(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    long_df = _explode_frames(df, id_col)
    if long_df.empty:
        return pd.DataFrame(columns=["frame", "count", "share"])
    dist = long_df["frame"].value_counts().reset_index()
    dist.columns = ["frame", "count"]
    dist["share"] = dist["count"] / max(1, dist["count"].sum())
    return dist


def run(ctx=None) -> dict[str, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("framing_analysis")
    videos, comments = _load_tables(ctx)
    outputs: dict[str, pd.DataFrame] = {}

    media_dist = _frame_distribution(videos, "video_id")
    public_dist = _frame_distribution(comments, "comment_id")
    if not media_dist.empty:
        save_table(media_dist, ctx.paths.tables / "media_frame_distribution.csv")
        outputs["media_frame_distribution"] = media_dist
    if not public_dist.empty:
        save_table(public_dist, ctx.paths.tables / "public_frame_distribution.csv")
        outputs["public_frame_distribution"] = public_dist

    all_frames = sorted(set(media_dist["frame"].dropna().tolist()) | set(public_dist["frame"].dropna().tolist()))
    if all_frames:
        media_vec = media_dist.set_index("frame")["count"].reindex(all_frames).fillna(0).to_numpy()
        public_vec = public_dist.set_index("frame")["count"].reindex(all_frames).fillna(0).to_numpy()
        gap = pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "js_divergence": js_divergence(media_vec, public_vec),
                    "media_total": float(media_vec.sum()),
                    "public_total": float(public_vec.sum()),
                }
            ]
        )
        save_table(gap, ctx.paths.tables / "media_public_framing_gap.csv")
        outputs["media_public_framing_gap"] = gap

    if not comments.empty and "stance" in comments.columns and "frames" in comments.columns:
        heat = comments[["stance", "frames"]].dropna().copy()
        heat["frames"] = heat["frames"].apply(listify_cell)
        heat = heat.explode("frames")
        if not heat.empty:
            matrix = heat.pivot_table(index="frames", columns="stance", values="frames", aggfunc="count", fill_value=0)
            save_table(matrix.reset_index(), ctx.paths.tables / "frame_stance_heatmap_framing.csv")
            if not matrix.empty:
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.heatmap(matrix, cmap="Purples", ax=ax)
                ax.set_title("Frame x Stance Heatmap")
                fig.tight_layout()
                fig.savefig(ctx.paths.figures / "frame_stance_heatmap_framing.png", dpi=180)
                plt.close(fig)

    logger.info("framing analysis complete videos=%s comments=%s", len(videos), len(comments))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run media-public framing comparison.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
