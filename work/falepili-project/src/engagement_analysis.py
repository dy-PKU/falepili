from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

from .common import ensure_dirs, listify_cell, load_context, load_table, save_table, setup_logging


STANCE_EXTREMITY = {
    "Strong Support": 2,
    "Support": 1,
    "Neutral": 0,
    "Concern": 1,
    "Strong Opposition": 2,
    "Unclear": 0,
}


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


def _build_model_frame(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return comments
    df = comments.copy()
    if not videos.empty:
        df = df.merge(
            videos[["video_id", "channel_name", "view_count", "like_count", "published_at"]].rename(
                columns={"like_count": "video_like_count", "view_count": "video_view_count", "published_at": "video_published_at"}
            ),
            on="video_id",
            how="left",
        )
    thread_reply_counts = df.groupby("thread_id")["comment_id"].transform(lambda s: max(0, len(s) - 1)) if "thread_id" in df.columns else 0
    df["reply_count"] = thread_reply_counts
    df["stance_extremity"] = df.get("stance", pd.Series(dtype=str)).map(STANCE_EXTREMITY).fillna(0)
    if "frames" in df.columns:
        df["frame_count"] = df["frames"].apply(lambda x: len(listify_cell(x)))
    else:
        df["frame_count"] = 0
    df["log_video_view_count"] = np.log1p(pd.to_numeric(df.get("video_view_count"), errors="coerce").fillna(0))
    df["log_comment_likes"] = np.log1p(pd.to_numeric(df.get("like_count"), errors="coerce").fillna(0))
    df["publication_month"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str)
    df["comment_age_days"] = pd.to_numeric(df.get("comment_age_days"), errors="coerce").fillna(0)
    return df


def _fit_nb_model(df: pd.DataFrame) -> Any:
    if df.empty:
        return None
    model_df = df[df["is_reply"] == False].copy() if "is_reply" in df.columns else df.copy()
    if len(model_df) < 10:
        return None
    formula = "reply_count ~ stance_extremity + frame_count + comment_age_days + log_video_view_count + C(channel_name)"
    try:
        return smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    except Exception:
        return None


def _fit_mixed_model(df: pd.DataFrame) -> Any:
    if df.empty:
        return None
    model_df = df[df["is_reply"] == False].copy() if "is_reply" in df.columns else df.copy()
    if len(model_df) < 10:
        return None
    try:
        return smf.mixedlm(
            "np.log1p(reply_count) ~ stance_extremity + frame_count + comment_age_days + log_video_view_count",
            data=model_df,
            groups=model_df["channel_name"].fillna("Unknown"),
        ).fit(reml=False, method="lbfgs")
    except Exception:
        return None


def _fit_like_model(df: pd.DataFrame) -> Any:
    if df.empty:
        return None
    model_df = df[df["is_reply"] == False].copy() if "is_reply" in df.columns else df.copy()
    if len(model_df) < 10:
        return None
    try:
        return smf.ols(
            "log_comment_likes ~ stance_extremity + frame_count + comment_age_days + log_video_view_count + C(channel_name)",
            data=model_df,
        ).fit(cov_type="HC1")
    except Exception:
        return None


def _coef_table(result: Any, model_name: str) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    params = getattr(result, "params", pd.Series(dtype=float))
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
    return pd.DataFrame(
        {
            "term": params.index.astype(str),
            "estimate": params.values,
            "p_value": pvalues.reindex(params.index).values,
            "model": model_name,
        }
    )


def run(ctx=None) -> dict[str, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("engagement_analysis")
    videos, comments = _load_inputs(ctx)
    df = _build_model_frame(videos, comments)
    outputs: dict[str, pd.DataFrame] = {}
    if df.empty:
        logger.warning("no data available for engagement analysis")
        return outputs

    nb = _fit_nb_model(df)
    mixed = _fit_mixed_model(df)
    like = _fit_like_model(df)

    coef_tables = pd.concat(
        [
            _coef_table(nb, "reply_nb"),
            _coef_table(mixed, "reply_mixed"),
            _coef_table(like, "like_ols"),
        ],
        ignore_index=True,
    )
    if not coef_tables.empty:
        save_table(coef_tables, ctx.paths.tables / "engagement_model_coefficients.csv")
        outputs["engagement_model_coefficients"] = coef_tables
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_df = coef_tables[coef_tables["term"].str.contains("stance_extremity|frame_count|comment_age_days|log_video_view_count", regex=True, na=False)]
        if not plot_df.empty:
            sns.barplot(data=plot_df, x="term", y="estimate", hue="model", ax=ax)
            ax.tick_params(axis="x", rotation=35)
            ax.set_title("Engagement Model Coefficients")
            fig.tight_layout()
            fig.savefig(ctx.paths.figures / "engagement_model_coefficients.png", dpi=180)
            plt.close(fig)

    model_text = []
    for name, result in [("Negative Binomial", nb), ("MixedLM", mixed), ("Like OLS", like)]:
        if result is not None:
            model_text.append(f"## {name}\n")
            model_text.append(str(result.summary()))
            model_text.append("\n")
    (ctx.paths.models / "engagement_models.txt").write_text("\n".join(model_text), encoding="utf-8")
    logger.info("engagement analysis complete rows=%s", len(df))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engagement regression analyses.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
