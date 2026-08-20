from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from . import __version__
from .common import load_context, setup_logging, utc_now_iso
from . import (
    descriptive_analysis,
    engagement_analysis,
    framing_analysis,
    llm_coding,
    normalize,
    relevance,
    temporal_analysis,
    validation,
    youtube_comments,
    youtube_search,
)


PIPELINE_STEPS: list[tuple[str, Callable[..., object]]] = [
    ("search", youtube_search.run),
    ("comments", youtube_comments.run),
    ("normalize", normalize.run),
    ("relevance", relevance.run),
    ("coding", llm_coding.run),
    ("validation", validation.run),
    ("descriptive", descriptive_analysis.run),
    ("temporal", temporal_analysis.run),
    ("framing", framing_analysis.run),
    ("engagement", engagement_analysis.run),
]


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"steps": {}, "updated_at": None, "version": __version__}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"steps": {}, "updated_at": None, "version": __version__}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def run_pipeline(demo: bool = False, force: bool = False) -> dict[str, object]:
    ctx = load_context()
    logger = setup_logging("pipeline")
    state_path = ctx.paths.interim / "pipeline_state.json"
    state = _load_state(state_path)
    results: dict[str, object] = {}
    for step_name, step_fn in PIPELINE_STEPS:
        if not force and state.get("steps", {}).get(step_name, {}).get("status") == "done":
            logger.info("skipping step=%s status=done", step_name)
            continue
        logger.info("running step=%s demo=%s", step_name, demo)
        if step_name in {"search", "comments"}:
            result = step_fn(ctx=ctx, demo=demo)
        else:
            result = step_fn(ctx=ctx)
        results[step_name] = result
        state.setdefault("steps", {})[step_name] = {"status": "done", "completed_at": utc_now_iso()}
        state["updated_at"] = utc_now_iso()
        state["version"] = __version__
        _save_state(state_path, state)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Falepili Union data pipeline.")
    parser.add_argument("--demo", action="store_true", help="Use deterministic synthetic data if API keys are absent.")
    parser.add_argument("--force", action="store_true", help="Re-run every step even if the state file marks it complete.")
    args = parser.parse_args()
    run_pipeline(demo=args.demo, force=args.force)


if __name__ == "__main__":
    main()

