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
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .common import ensure_dirs, listify_cell, load_context, load_table, save_table, setup_logging


STANCE_SCORE = {
    "Strong Support": 2,
    "Support": 1,
    "Neutral": 0,
    "Concern": -1,
    "Strong Opposition": -2,
    "Unclear": 0,
}
STANCE_EXTREMITY = {
    "Strong Support": 2,
    "Support": 1,
    "Neutral": 0,
    "Concern": 1,
    "Strong Opposition": 2,
    "Unclear": 0,
}


def _load_first(ctx, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = ctx.paths.processed / name
        if path.exists():
            return load_table(path)
    return pd.DataFrame()


def _load_inputs(ctx) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    post_names = [
        "reddit_posts_coded.parquet",
        "reddit_posts_coded_heuristic.parquet",
        "reddit_posts.parquet",
    ]
    comment_names = [
        "reddit_comments_coded.parquet",
        "reddit_comments_coded_heuristic.parquet",
        "reddit_comments.parquet",
    ]
    posts = _load_first(ctx, post_names)
    comments = _load_first(ctx, comment_names)
    source = "uncoded"
    for df in [posts, comments]:
        if not df.empty and "llm_source" in df.columns:
            source = str(df["llm_source"].fillna("unknown").mode().iloc[0])
            break
    return posts, comments, source


def _normalise_inputs(posts: pd.DataFrame, comments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = posts.copy()
    comments = comments.copy()
    if not posts.empty:
        posts["created_at"] = pd.to_datetime(posts.get("created_at"), utc=True, errors="coerce")
        posts["score"] = pd.to_numeric(posts.get("score"), errors="coerce").fillna(0)
        posts["num_comments"] = pd.to_numeric(posts.get("num_comments"), errors="coerce").fillna(0)
        posts["stance_score"] = posts.get("stance", pd.Series(index=posts.index, dtype=str)).map(STANCE_SCORE).fillna(0)
        posts["stance_extremity"] = posts.get("stance", pd.Series(index=posts.index, dtype=str)).map(STANCE_EXTREMITY).fillna(0)
        posts["frame_count"] = posts.get("frames", pd.Series(index=posts.index, dtype=object)).apply(lambda x: len(listify_cell(x)))
        posts["post_age_days"] = (pd.Timestamp.now(tz="UTC") - posts["created_at"]).dt.days.fillna(0)
    if not comments.empty:
        comments["created_at"] = pd.to_datetime(comments.get("created_at"), utc=True, errors="coerce")
        comments["score"] = pd.to_numeric(comments.get("score"), errors="coerce").fillna(0)
        comments["depth"] = pd.to_numeric(comments.get("depth"), errors="coerce").fillna(0)
        comments["stance_score"] = comments.get("stance", pd.Series(index=comments.index, dtype=str)).map(STANCE_SCORE).fillna(0)
        comments["stance_extremity"] = comments.get("stance", pd.Series(index=comments.index, dtype=str)).map(STANCE_EXTREMITY).fillna(0)
        comments["frame_count"] = comments.get("frames", pd.Series(index=comments.index, dtype=object)).apply(lambda x: len(listify_cell(x)))
        if not posts.empty and "subreddit" not in comments.columns:
            comments = comments.merge(posts[["reddit_post_id", "subreddit", "title"]], on="reddit_post_id", how="left")
    return posts, comments


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, ax=ax, color="#35605a")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _distribution(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[label, "count", "share"])
    out = df[col].fillna("Unknown").value_counts().reset_index()
    out.columns = [label, "count"]
    out["share"] = out["count"] / max(1, out["count"].sum())
    return out


def _frame_distribution(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty or "frames" not in df.columns:
        return pd.DataFrame(columns=["frame", "count", "share"])
    rows = []
    for _, row in df[[id_col, "frames"]].iterrows():
        for frame in listify_cell(row["frames"]):
            rows.append({id_col: row[id_col], "frame": frame})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame(columns=["frame", "count", "share"])
    out = long_df["frame"].value_counts().reset_index()
    out.columns = ["frame", "count"]
    out["share"] = out["count"] / max(1, out["count"].sum())
    return out


def _monthly_trends(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    source = comments if not comments.empty else posts
    id_col = "reddit_comment_id" if not comments.empty else "reddit_post_id"
    if source.empty or "created_at" not in source.columns:
        return pd.DataFrame()
    df = source.dropna(subset=["created_at"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["month"] = df["created_at"].dt.tz_convert(None).dt.to_period("M").astype(str)
    df["support"] = df.get("stance", pd.Series(index=df.index, dtype=str)).isin(["Strong Support", "Support"])
    df["opposition"] = df.get("stance", pd.Series(index=df.index, dtype=str)).isin(["Strong Opposition", "Concern"])
    out = df.groupby("month").agg(
        discussion_volume=(id_col, "count"),
        support_ratio=("support", "mean"),
        opposition_ratio=("opposition", "mean"),
        avg_stance_score=("stance_score", "mean"),
        avg_score=("score", "mean"),
    ).reset_index()
    out["month_ts"] = pd.to_datetime(out["month"] + "-01", utc=True)
    return out


def _event_study(source: pd.DataFrame, events: list[dict[str, Any]], id_col: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    windows: dict[str, pd.DataFrame] = {}
    if source.empty or "created_at" not in source.columns:
        return pd.DataFrame(), windows
    df = source.dropna(subset=["created_at"]).copy()
    for event in events:
        event_ts = pd.Timestamp(event["date"], tz="UTC")
        window = df[(df["created_at"] >= event_ts - pd.Timedelta(days=90)) & (df["created_at"] <= event_ts + pd.Timedelta(days=90))].copy()
        if window.empty:
            rows.append({"event": event["name"], "date": event["date"], "status": "insufficient_data"})
            continue
        window["relative_day"] = (window["created_at"] - event_ts).dt.days
        window["post"] = (window["relative_day"] >= 0).astype(int)
        daily = window.groupby("relative_day").agg(
            volume=(id_col, "count"),
            avg_stance_score=("stance_score", "mean"),
        ).reset_index()
        daily["post"] = (daily["relative_day"] >= 0).astype(int)
        daily["time_after"] = daily["relative_day"].clip(lower=0)
        windows[event["date"]] = daily.assign(event=event["name"])
        if daily["volume"].sum() < 20 or daily["post"].nunique() < 2:
            rows.append({"event": event["name"], "date": event["date"], "status": "insufficient_data"})
            continue
        try:
            model = smf.ols("avg_stance_score ~ relative_day + post + time_after", data=daily).fit(cov_type="HC1")
            rows.append(
                {
                    "event": event["name"],
                    "date": event["date"],
                    "status": "ok",
                    "params": json.dumps(model.params.to_dict(), ensure_ascii=False),
                    "pvalues": json.dumps(model.pvalues.to_dict(), ensure_ascii=False),
                    "rsquared": float(model.rsquared),
                }
            )
        except Exception as exc:
            rows.append({"event": event["name"], "date": event["date"], "status": "failed", "error": str(exc)})
    return pd.DataFrame(rows), windows


def _coef_table(result: Any, model_name: str) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame(columns=["term", "estimate", "p_value", "model"])
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


def _fit_models(posts: pd.DataFrame, comments: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model_text: list[str] = []
    tables: list[pd.DataFrame] = []
    if len(posts) >= 8 and "num_comments" in posts.columns:
        model_df = posts.copy()
        model_df["log_score"] = np.log1p(model_df["score"].clip(lower=0))
        try:
            nb = smf.glm(
                "num_comments ~ stance_extremity + frame_count + log_score + post_age_days",
                data=model_df,
                family=sm.families.NegativeBinomial(alpha=1.0),
            ).fit()
            tables.append(_coef_table(nb, "post_comment_count_nb"))
            model_text.extend(["## Post-Level Negative Binomial", str(nb.summary()), ""])
        except Exception as exc:
            model_text.append(f"Post-level NB failed: {exc}")
    if len(comments) >= 10:
        model_df = comments.copy()
        model_df["log_comment_score"] = np.log1p(model_df["score"].clip(lower=0))
        formula = "log_comment_score ~ stance_extremity + frame_count + depth"
        if "subreddit" in model_df.columns and model_df["subreddit"].nunique() > 1:
            formula += " + C(subreddit)"
        try:
            ols = smf.ols(formula, data=model_df).fit(cov_type="HC1")
            tables.append(_coef_table(ols, "comment_score_ols"))
            model_text.extend(["## Comment-Level Robust OLS", str(ols.summary()), ""])
        except Exception as exc:
            model_text.append(f"Comment-level OLS failed: {exc}")
    coef = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["term", "estimate", "p_value", "model"])
    if not model_text:
        model_text.append("Insufficient Reddit sample size for regression models.")
    return coef, "\n".join(model_text)


def _network_summary(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return pd.DataFrame(columns=["reddit_post_id", "comments", "unique_authors", "top_level_comments", "reply_edges", "max_depth"])
    df = comments.copy()
    top_level = df["parent_id"].astype(str).str.startswith("t3_", na=False) if "parent_id" in df.columns else pd.Series(False, index=df.index)
    out = df.groupby("reddit_post_id").agg(
        comments=("reddit_comment_id", "count"),
        unique_authors=("author", lambda s: int(s.dropna().nunique())),
        max_depth=("depth", "max"),
    ).reset_index()
    top = df.assign(top_level=top_level).groupby("reddit_post_id")["top_level"].sum().reset_index(name="top_level_comments")
    out = out.merge(top, on="reddit_post_id", how="left")
    out["reply_edges"] = out["comments"] - out["top_level_comments"].fillna(0)
    if not posts.empty:
        out = out.merge(posts[["reddit_post_id", "subreddit", "title"]], on="reddit_post_id", how="left")
    return out


def run(ctx=None) -> dict[str, pd.DataFrame]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("reddit_analysis")
    posts, comments, coding_source = _load_inputs(ctx)
    posts, comments = _normalise_inputs(posts, comments)
    outputs: dict[str, pd.DataFrame] = {}

    summary = pd.DataFrame(
        [
            {"metric": "reddit_posts", "value": len(posts)},
            {"metric": "reddit_comments", "value": len(comments)},
            {"metric": "coded_posts", "value": int("stance" in posts.columns) * len(posts)},
            {"metric": "coded_comments", "value": int("stance" in comments.columns) * len(comments)},
            {"metric": "coding_source", "value": coding_source},
        ]
    )
    save_table(summary, ctx.paths.tables / "reddit_summary_counts.csv")
    outputs["reddit_summary_counts"] = summary

    for df, prefix, id_col in [(posts, "reddit_post", "reddit_post_id"), (comments, "reddit_comment", "reddit_comment_id")]:
        stance = _distribution(df, "stance", "stance")
        emotion = _distribution(df, "emotion", "emotion")
        narrative = _distribution(df, "narrative_frame", "narrative_frame")
        frames = _frame_distribution(df, id_col)
        save_table(stance, ctx.paths.tables / f"{prefix}_stance_distribution.csv")
        save_table(emotion, ctx.paths.tables / f"{prefix}_emotion_distribution.csv")
        save_table(narrative, ctx.paths.tables / f"{prefix}_narrative_distribution.csv")
        save_table(frames, ctx.paths.tables / f"{prefix}_frame_prevalence.csv")
        _plot_bar(stance, "stance", "count", ctx.paths.figures / f"{prefix}_stance_distribution.png", f"{prefix} Stance")
        _plot_bar(frames.head(12), "frame", "count", ctx.paths.figures / f"{prefix}_frame_prevalence.png", f"{prefix} Frames")
        outputs[f"{prefix}_stance_distribution"] = stance
        outputs[f"{prefix}_frame_prevalence"] = frames

    subreddit = _distribution(posts, "subreddit", "subreddit")
    save_table(subreddit, ctx.paths.tables / "reddit_subreddit_distribution.csv")
    _plot_bar(subreddit.head(15), "subreddit", "count", ctx.paths.figures / "reddit_subreddit_distribution.png", "Reddit Subreddits")
    outputs["reddit_subreddit_distribution"] = subreddit

    monthly = _monthly_trends(posts, comments)
    if not monthly.empty:
        save_table(monthly, ctx.paths.tables / "reddit_monthly_trends.csv")
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=monthly, x="month_ts", y="discussion_volume", marker="o", ax=ax)
        for event in ctx.events_config.get("events", []):
            ax.axvline(pd.Timestamp(event["date"], tz="UTC"), color="grey", alpha=0.25)
        ax.set_title("Reddit Monthly Discussion Volume")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(ctx.paths.figures / "reddit_monthly_discussion_volume.png", dpi=180)
        plt.close(fig)
        outputs["reddit_monthly_trends"] = monthly

    event_source = comments if not comments.empty else posts
    event_id = "reddit_comment_id" if not comments.empty else "reddit_post_id"
    event_df, windows = _event_study(event_source, ctx.events_config.get("events", []), event_id)
    save_table(event_df, ctx.paths.tables / "reddit_event_study_summary.csv")
    outputs["reddit_event_study_summary"] = event_df
    for date, window in windows.items():
        save_table(window, ctx.paths.tables / f"reddit_event_window_{date}.csv")

    coef, model_text = _fit_models(posts, comments)
    save_table(coef, ctx.paths.tables / "reddit_engagement_model_coefficients.csv")
    (ctx.paths.models / "reddit_models.txt").write_text(model_text, encoding="utf-8")
    outputs["reddit_engagement_model_coefficients"] = coef

    network = _network_summary(posts, comments)
    save_table(network, ctx.paths.tables / "reddit_thread_network_summary.csv")
    outputs["reddit_thread_network_summary"] = network

    logger.info("reddit analysis complete posts=%s comments=%s coding_source=%s", len(posts), len(comments), coding_source)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reddit-only computational social science analyses.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
