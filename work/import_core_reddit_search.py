from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common import append_jsonl, load_context, save_table, utc_now_iso
from src.reddit_json import _is_direct_post_relevant, _redditapis_post_from_item


ROOT = Path.cwd()
WORK = ROOT.parents[2]
SOURCES = [
    (WORK / "work/redditapis-australia-tuvalu.json", "Tuvalu", "australia"),
    (WORK / "work/redditapis-australia-climate-migration.json", "climate migration", "australia"),
    (WORK / "work/redditapis-australianpolitics-tuvalu.json", "Tuvalu", "AustralianPolitics"),
    (WORK / "work/redditapis-australianpolitics-climate-migration.json", "climate migration", "AustralianPolitics"),
]

ctx = load_context()
retrieved_at = utc_now_iso()
rows = []
for path, keyword, subreddit in SOURCES:
    payload = json.loads(path.read_text(encoding="utf-8"))
    append_jsonl(
        ctx.paths.raw / "reddit_search_responses.jsonl",
        {
            "retrieved_at": retrieved_at,
            "keyword": keyword,
            "subreddit": subreddit,
            "page": 0,
            "after": None,
            "url": "https://api.redditapis.com/api/reddit/search",
            "params": {"q": keyword, "subreddit": subreddit, "sort": "new", "time_filter": "all", "limit": 100},
            "payload": payload,
            "transport": "curl_import",
        },
    )
    for item in payload.get("posts", []):
        row = _redditapis_post_from_item(item, keyword, subreddit, retrieved_at)
        text = f"{row.get('title') or ''} {row.get('selftext') or ''}"
        relevant = _is_direct_post_relevant(text)
        row["relevance_status"] = "directly relevant" if relevant else "irrelevant"
        row["relevance_score"] = 1 if relevant else 0
        created = float(row.get("created_utc") or 0)
        cutoff = datetime(2023, 11, 9, tzinfo=timezone.utc).timestamp()
        if relevant and created >= cutoff:
            rows.append(row)

existing_path = ctx.paths.processed / "reddit_posts.parquet"
existing = pd.read_parquet(existing_path) if existing_path.exists() else pd.DataFrame()
new = pd.DataFrame(rows)
combined = pd.concat([existing, new], ignore_index=True, sort=False)
combined = combined.drop_duplicates(subset=["reddit_post_id"], keep="last")
save_table(combined, existing_path)

core = combined[combined["subreddit_scope"].isin(["australia", "AustralianPolitics"])].copy()
manifest = core[["reddit_post_id", "subreddit", "title", "permalink", "num_comments", "score"]]
manifest.to_csv(WORK / "work/core_reddit_posts_manifest.csv", index=False)
print(json.dumps({"new_relevant_rows": len(new), "combined_posts": len(combined), "core_posts": len(core)}, ensure_ascii=False))
