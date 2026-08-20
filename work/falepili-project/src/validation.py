from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

from .common import ensure_dirs, load_context, load_table, normalize_text, save_table, setup_logging


def krippendorff_alpha_nominal(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy()
    values = values[~pd.isna(values).all(axis=1)]
    if values.size == 0:
        return float("nan")
    categories = pd.unique(values.ravel("K"))
    categories = [cat for cat in categories if pd.notna(cat)]
    if not categories:
        return float("nan")
    coincidence = {cat: {other: 0 for other in categories} for cat in categories}
    for row in values:
        row = [item for item in row if pd.notna(item)]
        if len(row) < 2:
            continue
        for i, a in enumerate(row):
            for b in row[i + 1 :]:
                coincidence[a][b] += 1
                coincidence[b][a] += 1
                coincidence[a][a] += 1
                coincidence[b][b] += 1
    total_pairs = sum(sum(inner.values()) for inner in coincidence.values())
    if total_pairs == 0:
        return float("nan")
    do = 0.0
    for cat_a, row in coincidence.items():
        for cat_b, count in row.items():
            if cat_a != cat_b:
                do += count
    do = do / total_pairs
    marginals = {cat: sum(row.values()) for cat, row in coincidence.items()}
    de = 0.0
    for cat_a, count_a in marginals.items():
        for cat_b, count_b in marginals.items():
            if cat_a != cat_b:
                de += count_a * count_b
    total = sum(marginals.values())
    if total <= 1:
        return float("nan")
    de = de / (total * (total - 1))
    return float(1 - do / de) if de else float("nan")


def _multi_label_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    classes = sorted({label for cell in y_true.dropna() for label in _split_labels(cell)} | {label for cell in y_pred.dropna() for label in _split_labels(cell)})
    if not classes:
        return {"precision_macro": float("nan"), "recall_macro": float("nan"), "f1_macro": float("nan")}
    y_true_bin = _binarize(y_true, classes)
    y_pred_bin = _binarize(y_pred, classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    return {"precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)}


def _split_labels(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.split(";") if part.strip()]


def _binarize(series: pd.Series, classes: list[str]) -> np.ndarray:
    matrix = np.zeros((len(series), len(classes)), dtype=int)
    index = {label: idx for idx, label in enumerate(classes)}
    for row_idx, cell in enumerate(series.fillna("")):
        for label in _split_labels(cell):
            if label in index:
                matrix[row_idx, index[label]] = 1
    return matrix


def export_annotation_sample(ctx, n: int = 1000) -> pd.DataFrame:
    comments_path = ctx.paths.processed / "comments_coded.parquet"
    comments = load_table(comments_path) if comments_path.exists() else load_table(ctx.paths.processed / "comments.parquet")
    if comments.empty:
        return comments
    if "llm_relevance" in comments.columns:
        relevance_series = comments["llm_relevance"]
    elif "relevance_status" in comments.columns:
        relevance_series = comments["relevance_status"]
    else:
        relevance_series = pd.Series(index=comments.index, dtype=str)
    relevant = comments[relevance_series.isin(["directly relevant", "contextually relevant"])]
    pool = relevant if not relevant.empty else comments
    sample_n = min(n, len(pool))
    sample = pool.sample(sample_n, random_state=ctx.random_seed).copy()
    sample["human_stance"] = ""
    sample["human_frames"] = ""
    sample["human_emotion"] = ""
    sample["human_narrative_frame"] = ""
    sample["human_relevance"] = ""
    save_table(sample, ctx.paths.gold_standard / "annotation_sample.csv")
    template = sample[["comment_id", "video_id", "text", "llm_relevance", "stance", "frames", "emotion", "narrative_frame"]].copy()
    template.to_csv(ctx.paths.gold_standard / "annotation_template.csv", index=False)
    return sample


def _maybe_load_human_annotations(ctx) -> pd.DataFrame:
    path = ctx.paths.gold_standard / "human_annotations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def compute_metrics(reference: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    merged = reference.merge(prediction, on="comment_id", suffixes=("_ref", "_pred"))
    metrics: dict[str, Any] = {"n": int(len(merged))}
    if merged.empty:
        return metrics
    if {"human_stance", "stance_pred"}.issubset(merged.columns):
        stance_valid = merged[["human_stance", "stance_pred"]].dropna()
        if not stance_valid.empty:
            metrics["stance_accuracy"] = float(accuracy_score(stance_valid["human_stance"], stance_valid["stance_pred"]))
            metrics["stance_kappa"] = float(cohen_kappa_score(stance_valid["human_stance"], stance_valid["stance_pred"]))
    if {"human_emotion", "emotion_pred"}.issubset(merged.columns):
        emotion_valid = merged[["human_emotion", "emotion_pred"]].dropna()
        if not emotion_valid.empty:
            metrics["emotion_accuracy"] = float(accuracy_score(emotion_valid["human_emotion"], emotion_valid["emotion_pred"]))
            metrics["emotion_kappa"] = float(cohen_kappa_score(emotion_valid["human_emotion"], emotion_valid["emotion_pred"]))
    if {"human_frames", "frames_pred"}.issubset(merged.columns):
        metrics.update({f"frames_{k}": v for k, v in _multi_label_metrics(merged["human_frames"], merged["frames_pred"]).items()})
    if {"human_relevance", "llm_relevance_pred"}.issubset(merged.columns):
        rel_valid = merged[["human_relevance", "llm_relevance_pred"]].dropna()
        if not rel_valid.empty:
            metrics["relevance_accuracy"] = float(accuracy_score(rel_valid["human_relevance"], rel_valid["llm_relevance_pred"]))
            metrics["relevance_kappa"] = float(cohen_kappa_score(rel_valid["human_relevance"], rel_valid["llm_relevance_pred"]))
    if {"human_stance", "stance_pred"}.issubset(merged.columns):
        alpha_df = merged[["human_stance", "stance_pred"]].rename(columns={"human_stance": "coder_a", "stance_pred": "coder_b"})
        metrics["krippendorff_alpha_stance"] = float(krippendorff_alpha_nominal(alpha_df))
    return metrics


def run(ctx=None, sample_n: int = 1000) -> dict[str, Any]:
    ctx = ctx or load_context()
    ensure_dirs(ctx.paths)
    logger = setup_logging("validation")
    sample = export_annotation_sample(ctx, n=sample_n)
    human = _maybe_load_human_annotations(ctx)
    if human.empty:
        metrics = {"n_sample": int(len(sample)), "status": "sample_exported"}
        (ctx.paths.reports / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("exported annotation sample=%s", len(sample))
        return metrics
    llm = load_table(ctx.paths.processed / "comments_coded.parquet") if (ctx.paths.processed / "comments_coded.parquet").exists() else pd.DataFrame()
    if llm.empty:
        raise FileNotFoundError("comments_coded.parquet is required for validation metrics.")
    ref = human.copy()
    pred = llm[["comment_id", "stance", "frames", "emotion", "narrative_frame", "llm_relevance"]].copy()
    pred.columns = ["comment_id", "stance_pred", "frames_pred", "emotion_pred", "narrative_frame_pred", "llm_relevance_pred"]
    metrics = compute_metrics(ref, pred)
    (ctx.paths.reports / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("computed validation metrics n=%s", metrics.get("n"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation sample and compute agreement metrics.")
    parser.add_argument("--sample-n", type=int, default=1000, help="Number of comments to sample for human coding.")
    args = parser.parse_args()
    run(sample_n=args.sample_n)


if __name__ == "__main__":
    main()
