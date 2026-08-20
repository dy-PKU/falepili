from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import ensure_dirs, explode_multilabel, listify_cell, load_context, load_table, save_table, setup_logging


def _load_inputs(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_path = ctx.paths.processed / "videos_coded.parquet"
    comments_path = ctx.paths.processed / "comments_coded.parquet"
    if not videos_path.exists():
        videos_path = ctx.paths.processed / "videos_relevance.parquet"
    if not comments_path.exists():
        comments_path = ctx.paths.processed / "comments_relevance.parquet"
    videos = load_table(videos_path) if videos_path.exists() else pd.DataFrame()
    comments = load_table(comments_path) if comments_path.exists() else pd.DataFrame()
    return videos, comments


def _pick_series(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns:
        return df[preferred]
    if fallback in df.columns:
        return df[fallback]
    return pd.Series(dtype=str)


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, ax=ax, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(ctx=None) -> dict[str, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("descriptive_analysis")
    videos, comments = _load_inputs(ctx)
    outputs: dict[str, pd.DataFrame] = {}

    if comments.empty:
        top_level = 0
        relevant = 0
    else:
        is_reply = comments["is_reply"] if "is_reply" in comments.columns else pd.Series(False, index=comments.index)
        relevance_series = comments["llm_relevance"] if "llm_relevance" in comments.columns else comments["relevance_status"] if "relevance_status" in comments.columns else pd.Series(dtype=str)
        top_level = int((~is_reply.astype(bool)).sum())
        relevant = int(relevance_series.isin(["directly relevant", "contextually relevant"]).sum())
    counts = pd.DataFrame(
        [
            {"metric": "videos", "value": len(videos)},
            {"metric": "comments", "value": len(comments)},
            {"metric": "top_level_comments", "value": top_level},
            {"metric": "relevant_comments", "value": relevant},
        ]
    )
    save_table(counts, ctx.paths.tables / "summary_counts.csv")
    outputs["summary_counts"] = counts

    if not comments.empty:
        stance_col = "stance" if "stance" in comments.columns else "relevance_status"
        stance = comments[stance_col].fillna("Unclear").value_counts(dropna=False).reset_index()
        stance.columns = ["stance", "count"]
        stance["share"] = stance["count"] / max(1, stance["count"].sum())
        save_table(stance, ctx.paths.tables / "stance_distribution.csv")
        _plot_bar(stance, "stance", "count", ctx.paths.figures / "stance_distribution.png", "Stance Distribution")
        outputs["stance_distribution"] = stance

        emotion = comments.get("emotion", pd.Series(dtype=str)).fillna("Neutral").value_counts(dropna=False).reset_index()
        emotion.columns = ["emotion", "count"]
        emotion["share"] = emotion["count"] / max(1, emotion["count"].sum())
        save_table(emotion, ctx.paths.tables / "emotion_distribution.csv")
        _plot_bar(emotion, "emotion", "count", ctx.paths.figures / "emotion_distribution.png", "Emotion Distribution")
        outputs["emotion_distribution"] = emotion

        narrative = comments.get("narrative_frame", pd.Series(dtype=str)).fillna("Other").value_counts(dropna=False).reset_index()
        narrative.columns = ["narrative_frame", "count"]
        narrative["share"] = narrative["count"] / max(1, narrative["count"].sum())
        save_table(narrative, ctx.paths.tables / "narrative_distribution.csv")
        _plot_bar(narrative, "narrative_frame", "count", ctx.paths.figures / "narrative_distribution.png", "Narrative Frame Distribution")
        outputs["narrative_distribution"] = narrative

        frames_source = comments[["comment_id", "frames"]].copy() if "frames" in comments.columns else pd.DataFrame(columns=["comment_id", "frames"])
        frame_long = explode_multilabel(frames_source, "comment_id", "frames") if not frames_source.empty else pd.DataFrame(columns=["comment_id", "frames"])
        if not frame_long.empty:
            frame_dist = frame_long["frames"].value_counts().reset_index()
            frame_dist.columns = ["frame", "count"]
            frame_dist["share"] = frame_dist["count"] / max(1, frame_dist["count"].sum())
            save_table(frame_dist, ctx.paths.tables / "frame_prevalence.csv")
            _plot_bar(frame_dist, "frame", "count", ctx.paths.figures / "frame_prevalence.png", "Frame Prevalence")
            outputs["frame_prevalence"] = frame_dist

            if "stance" in comments.columns:
                cross = comments[["comment_id", "stance", "frames"]].dropna()
                cross = cross.assign(frames=cross["frames"].apply(listify_cell))
                cross = cross.explode("frames")
                heat = cross.pivot_table(index="frames", columns="stance", values="comment_id", aggfunc="count", fill_value=0)
                save_table(heat.reset_index(), ctx.paths.tables / "frame_stance_heatmap.csv")
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.heatmap(heat, cmap="Blues", ax=ax)
                ax.set_title("Frame x Stance Heatmap")
                fig.tight_layout()
                fig.savefig(ctx.paths.figures / "frame_stance_heatmap.png", dpi=180)
                plt.close(fig)
                outputs["frame_stance_heatmap"] = heat.reset_index()

    if not videos.empty:
        channel = videos["channel_name"].fillna("Unknown").value_counts().reset_index()
        channel.columns = ["channel_name", "count"]
        save_table(channel, ctx.paths.tables / "channel_distribution.csv")
        _plot_bar(channel.head(10), "channel_name", "count", ctx.paths.figures / "channel_distribution.png", "Top Channels")
        outputs["channel_distribution"] = channel

    logger.info("descriptive analysis complete videos=%s comments=%s", len(videos), len(comments))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptive analysis and export tables/figures.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
