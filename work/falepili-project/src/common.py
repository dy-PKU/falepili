from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import requests
import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(path: Path, override: bool = False, *args, **kwargs):  # type: ignore[misc]
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value
        return True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    raw: Path
    interim: Path
    processed: Path
    gold_standard: Path
    tables: Path
    figures: Path
    models: Path
    reports: Path
    logs: Path
    llm_cache: Path


@dataclass(frozen=True)
class ProjectContext:
    paths: ProjectPaths
    keywords_config: dict[str, Any]
    events_config: dict[str, Any]
    youtube_api_key: str | None
    openai_api_key: str | None
    openai_model: str
    llm_batch_size: int
    random_seed: int


def get_paths(root: Path = PROJECT_ROOT) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        config=root / "config",
        raw=root / "data" / "raw",
        interim=root / "data" / "interim",
        processed=root / "data" / "processed",
        gold_standard=root / "data" / "gold_standard",
        tables=root / "outputs" / "tables",
        figures=root / "outputs" / "figures",
        models=root / "outputs" / "models",
        reports=root / "outputs" / "reports",
        logs=root / "logs",
        llm_cache=root / "data" / "interim" / "llm_cache",
    )


def ensure_dirs(paths: ProjectPaths | None = None) -> None:
    paths = paths or get_paths()
    for folder in (
        paths.config,
        paths.raw,
        paths.interim,
        paths.processed,
        paths.gold_standard,
        paths.tables,
        paths.figures,
        paths.models,
        paths.reports,
        paths.logs,
        paths.llm_cache,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_context() -> ProjectContext:
    paths = get_paths()
    ensure_dirs(paths)
    load_dotenv(paths.root / ".env", override=True)
    youtube_api_key = os.getenv("YOUTUBE_API_KEY") or None
    youtube_api_key_path = os.getenv("YOUTUBE_API_KEY_PATH") or None
    if not youtube_api_key and youtube_api_key_path:
        key_path = Path(youtube_api_key_path)
        if not key_path.is_absolute():
            key_path = paths.root / key_path
        if key_path.exists():
            youtube_api_key = key_path.read_text(encoding="utf-8").strip() or None
    keywords_config = load_yaml_file(paths.config / "keywords.yaml")
    events_config = load_yaml_file(paths.config / "events.yaml")
    seed = int(keywords_config.get("study", {}).get("random_seed", 42))
    seed_everything(seed)
    return ProjectContext(
        paths=paths,
        keywords_config=keywords_config,
        events_config=events_config,
        youtube_api_key=youtube_api_key,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        llm_batch_size=int(os.getenv("LLM_BATCH_SIZE", "25")),
        random_seed=seed,
    )


def setup_logging(name: str) -> logging.Logger:
    paths = get_paths()
    ensure_dirs(paths)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(paths.logs / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now_iso() -> str:
    return dt.datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date(value: str | dt.date | None) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def iso_datetime_for_date(value: dt.date, end_of_day: bool = False) -> str:
    if end_of_day:
        value_dt = dt.datetime.combine(value, dt.time(23, 59, 59), tzinfo=UTC)
    else:
        value_dt = dt.datetime.combine(value, dt.time(0, 0, 0), tzinfo=UTC)
    return value_dt.isoformat().replace("+00:00", "Z")


def month_windows(start: dt.date, end: dt.date) -> Iterator[tuple[dt.date, dt.date]]:
    current = start
    while current <= end:
        if current.month == 12:
            next_month = dt.date(current.year + 1, 1, 1)
        else:
            next_month = dt.date(current.year, current.month + 1, 1)
        window_end = min(end, next_month - dt.timedelta(days=1))
        yield current, window_end
        current = next_month


def default_study_dates(ctx: ProjectContext) -> tuple[dt.date, dt.date]:
    study = ctx.keywords_config.get("study", {})
    start = parse_date(study.get("start_date")) or dt.date(2023, 11, 9)
    end = parse_date(study.get("end_date")) or dt.datetime.now(tz=UTC).date()
    return start, end


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chunked(values: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _redact_url_secrets(text: Any) -> str:
    return re.sub(r"([?&](?:key|access_token|token|api_key)=)[^&\s]+", r"\1***", str(text))


def request_json_with_backoff(
    url: str,
    params: dict[str, Any],
    logger: logging.Logger,
    timeout: int = 30,
    max_retries: int = 6,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(60, 2**attempt + random.random())
                logger.warning("retryable_http_status status=%s wait=%.1f url=%s", response.status_code, wait, _redact_url_secrets(response.url))
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                reason = _youtube_error_reason(payload)
                if reason in {"quotaExceeded", "rateLimitExceeded", "userRateLimitExceeded"}:
                    raise RuntimeError(f"YouTube quota/rate limit error: {reason}")
                raise RuntimeError(f"YouTube API error: {payload['error']}")
            return payload
        except requests.RequestException as exc:
            last_error = exc
            wait = min(60, 2**attempt + random.random())
            logger.warning("request_failed attempt=%s wait=%.1f error=%s", attempt + 1, wait, _redact_url_secrets(exc))
            time.sleep(wait)
    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")


def _youtube_error_reason(payload: dict[str, Any]) -> str | None:
    errors = payload.get("error", {}).get("errors", [])
    if errors:
        return errors[0].get("reason")
    return payload.get("error", {}).get("status")


def listify_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, (tuple, set, np.ndarray)):
        return [str(v) for v in list(value)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in value.split(";") if part.strip()]
    return [str(value)]


def explode_multilabel(df: pd.DataFrame, id_col: str, label_col: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        labels = listify_cell(row.get(label_col))
        for label in labels:
            records.append({id_col: row[id_col], label_col: label})
    return pd.DataFrame(records)


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.sum() == 0 or q_arr.sum() == 0:
        return float("nan")
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    midpoint = 0.5 * (p_arr + q_arr)
    return float(0.5 * _kl_div(p_arr, midpoint) + 0.5 * _kl_div(q_arr, midpoint))


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--demo", action="store_true", help="Use deterministic synthetic demo data.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
