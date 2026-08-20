from __future__ import annotations

import argparse
import json
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

from .common import ensure_dirs, load_context, load_table, save_table, setup_logging


STANCE_SCORE = {
    "Strong Support": 2,
    "Support": 1,
    "Neutral": 0,
    "Concern": -1,
    "Strong Opposition": -2,
    "Unclear": 0,
}


def _load_comments(ctx) -> pd.DataFrame:
    for candidate in ["comments_coded.parquet", "comments_relevance.parquet", "comments.parquet"]:
        path = ctx.paths.processed / candidate
        if path.exists():
            return load_table(path)
    return pd.DataFrame()


def _prepare_monthly_series(comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame()
    df = comments.copy()
    df["month"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str)
    df["stance_score"] = df.get("stance", pd.Series(dtype=str)).map(STANCE_SCORE).fillna(0)
    df["support"] = df["stance"].isin(["Strong Support", "Support"]) if "stance" in df.columns else False
    df["opposition"] = df["stance"].isin(["Strong Opposition", "Concern"]) if "stance" in df.columns else False
    grouped = df.groupby("month").agg(
        discussion_volume=("comment_id", "count"),
        support_ratio=("support", "mean"),
        opposition_ratio=("opposition", "mean"),
        avg_stance_score=("stance_score", "mean"),
        avg_likes=("like_count", "mean"),
    ).reset_index()
    grouped["month_ts"] = pd.to_datetime(grouped["month"] + "-01", utc=True, errors="coerce")
    return grouped


def _event_window(comments: pd.DataFrame, event_date: str, window_days: int = 90) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame()
    event_ts = pd.Timestamp(event_date, tz="UTC")
    df = comments.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df[(df["published_at"] >= event_ts - pd.Timedelta(days=window_days)) & (df["published_at"] <= event_ts + pd.Timedelta(days=window_days))].copy()
    if df.empty:
        return df
    df["relative_day"] = (df["published_at"] - event_ts).dt.days
    df["post"] = (df["relative_day"] >= 0).astype(int)
    df["stance_score"] = df.get("stance", pd.Series(dtype=str)).map(STANCE_SCORE).fillna(0)
    df["support"] = df["stance"].isin(["Strong Support", "Support"]) if "stance" in df.columns else False
    daily = df.groupby("relative_day").agg(
        volume=("comment_id", "count"),
        support_ratio=("support", "mean"),
        avg_stance_score=("stance_score", "mean"),
    ).reset_index()
    daily["post"] = (daily["relative_day"] >= 0).astype(int)
    daily["time_after"] = daily["relative_day"].clip(lower=0)
    return daily


def _fit_its(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or daily["volume"].sum() < 20:
        return {"status": "insufficient_data"}
    try:
        model = smf.ols("support_ratio ~ relative_day + post + time_after", data=daily).fit(cov_type="HC1")
        return {
            "status": "ok",
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "rsquared": float(model.rsquared),
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run(ctx=None) -> dict[str, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("temporal_analysis")
    comments = _load_comments(ctx)
    outputs: dict[str, pd.DataFrame] = {}
    monthly = _prepare_monthly_series(comments)
    if not monthly.empty:
        save_table(monthly, ctx.paths.tables / "monthly_trends.csv")
        outputs["monthly_trends"] = monthly
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=monthly, x="month_ts", y="discussion_volume", marker="o", ax=ax)
        for event in ctx.events_config.get("events", []):
            ax.axvline(pd.Timestamp(event["date"], tz="UTC"), color="grey", alpha=0.25)
        ax.set_title("Monthly Discussion Volume")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(ctx.paths.figures / "monthly_discussion_volume.png", dpi=180)
        plt.close(fig)

    event_rows = []
    for event in ctx.events_config.get("events", []):
        daily = _event_window(comments, event["date"])
        its = _fit_its(daily)
        row = {"event": event["name"], "date": event["date"], **its}
        event_rows.append(row)
        if not daily.empty:
            daily["event"] = event["name"]
            save_table(daily, ctx.paths.tables / f"event_window_{event['date']}.csv")
    event_df = pd.DataFrame(event_rows)
    if not event_df.empty:
        save_table(event_df, ctx.paths.tables / "event_study_summary.csv")
        outputs["event_study_summary"] = event_df
    logger.info("temporal analysis complete monthly=%s events=%s", len(monthly), len(event_df))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal and event-window analysis.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
