from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import load_context, stable_hash
from src.llm_coding import _build_openai_payload, _prompt_text

ctx = load_context()
if not ctx.openai_api_key:
    raise SystemExit("OPENAI_API_KEY is not configured")
root = ctx.paths.root
work = root.parents[2] / "work/openai-reddit-smoke"
work.mkdir(parents=True, exist_ok=True)
posts = pd.read_parquet(root / "data/processed/reddit_posts.parquet")
comments = pd.read_parquet(root / "data/processed/reddit_comments.parquet")
comments = comments[comments.get("content_status", "available") == "available"].copy()
context = posts[["reddit_post_id", "title", "subreddit"]].rename(
    columns={"title": "thread_title", "subreddit": "thread_subreddit"}
)
comments = comments.merge(context, on="reddit_post_id", how="left")
comments["subreddit"] = comments.get("subreddit", comments["thread_subreddit"]).fillna(comments["thread_subreddit"])
manifest = []
for i, (_, row) in enumerate(comments.head(10).iterrows()):
    prompt = _prompt_text(row, "reddit_comment")
    key = stable_hash({"kind": "reddit_comment", "prompt": prompt, "model": ctx.openai_model, "backend": "openai"})
    (work / f"request-{i:02d}.json").write_text(
        json.dumps(_build_openai_payload(prompt, ctx), ensure_ascii=False), encoding="utf-8"
    )
    manifest.append({"index": i, "comment_id": row["reddit_comment_id"], "cache_key": key, "row": row.to_dict()})
(work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, default=str), encoding="utf-8")
conf = Path("/private/tmp/falepili-openai-curl.conf")
conf.write_text(
    'silent\nshow-error\nmax-time = 120\n'
    f'header = "Authorization: Bearer {ctx.openai_api_key}"\n'
    'header = "Content-Type: application/json"\n', encoding="utf-8"
)
conf.chmod(0o600)
print(json.dumps({"requests": len(manifest), "model": ctx.openai_model, "work": str(work)}))
