from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import append_jsonl, load_context, save_table, utc_now_iso
from src.reddit_json import _comment_records_from_nodes, _is_direct_comment_relevant


ROOT = Path.cwd()
WORK = ROOT.parents[2]
ctx = load_context()
retrieved_at = utc_now_iso()
rows = []

for path in sorted((WORK / "work").glob("comments-*.json")):
    post_id = path.stem.removeprefix("comments-")
    payload = json.loads(path.read_text(encoding="utf-8"))
    append_jsonl(
        ctx.paths.raw / "reddit_comment_responses.jsonl",
        {
            "retrieved_at": retrieved_at,
            "post_id": post_id,
            "url": "https://api.redditapis.com/api/reddit/comments",
            "params": {"depth": 10, "limit": 500, "sort": "confidence"},
            "payload": payload,
            "transport": "curl_import",
        },
    )
    records = _comment_records_from_nodes(post_id, payload.get("comments", []), retrieved_at)
    for record in records:
        record["content_status"] = "available" if str(record.get("body") or "").strip() else "deleted_or_unavailable"
        relevant = _is_direct_comment_relevant(record.get("body"))
        record["relevance_status"] = "directly relevant" if relevant else "contextually relevant"
        record["relevance_score"] = 1 if relevant else 0
        record["thread_relevance_status"] = "directly relevant"
        rows.append(record)

existing_path = ctx.paths.processed / "reddit_comments.parquet"
existing = pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
if not existing.empty and "content_status" not in existing.columns:
    existing["content_status"] = existing["body"].fillna("").astype(str).str.strip().map(
        lambda value: "available" if value else "deleted_or_unavailable"
    )
new = pd.DataFrame(rows)
combined = pd.concat([existing, new], ignore_index=True, sort=False)
combined = combined.drop_duplicates(subset=["reddit_comment_id"], keep="last")
save_table(combined, existing_path)
print(json.dumps({"new_comment_rows": len(new), "combined_comments": len(combined)}, ensure_ascii=False))
