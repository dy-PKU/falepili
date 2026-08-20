# Falepili Project

Reproducible YouTube data pipeline for studying Australian digital public discourse about the Australia-Tuvalu Falepili Union, Tuvalu migration, climate mobility, and security cooperation.

## Run

```bash
pip install -r requirements.txt
copy .env.example .env
python -m src.main
```

The pipeline automatically falls back to deterministic demo data when `YOUTUBE_API_KEY` or `OPENAI_API_KEY` is missing.

## Inputs

- `config/keywords.yaml`
- `config/events.yaml`

## Outputs

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/gold_standard/`
- `outputs/tables/`
- `outputs/figures/`
- `outputs/models/`
- `outputs/reports/`

## Module entry points

```bash
python -m src.youtube_search
python -m src.youtube_comments
python -m src.reddit_json
python -m src.reddit_json --auth-mode redditapis
python -m src.reddit_json --auth-mode oauth --proxy-url http://127.0.0.1:10808
python -m src.normalize
python -m src.relevance
python -m src.llm_coding
python -m src.validation
python -m src.descriptive_analysis
python -m src.temporal_analysis
python -m src.framing_analysis
python -m src.engagement_analysis
```

## Data schema

### Videos

- `video_id`
- `title`
- `description`
- `channel_id`
- `channel_name`
- `published_at`
- `query_source`
- `search_window`
- `view_count`
- `like_count`
- `comment_count`
- `retrieved_at`
- `relevance_status`

### Comments

- `comment_id`
- `video_id`
- `parent_id`
- `thread_id`
- `text`
- `like_count`
- `published_at`
- `updated_at`
- `is_reply`
- `retrieved_at`

## Notes

- Raw API responses are stored as JSONL.
- Normalized tables are written to Parquet.
- LLM outputs are cached under `data/interim/llm_cache/`.
- `data/interim/pipeline_state.json` tracks completed steps for resumable runs.
- Reddit collection supports explicit proxies via `REDDIT_PROXY_URL` or `--proxy-url`.
- RedditAPIs mode reads `key.txt` by default through `config/reddit.yaml` and uses `REDDIT_AUTH_MODE=redditapis`.
- Reddit collection runs in strict relevance mode by default and keeps only directly on-topic posts and comments.
- Reddit collection can also use `curl_cffi` TLS impersonation via `config/reddit.yaml` or the `transport` setting.
- Reddit OAuth mode uses `oauth.reddit.com` with a bearer token. Set `REDDIT_AUTH_MODE=oauth` and either `REDDIT_ACCESS_TOKEN`, or `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` with `REDDIT_GRANT_TYPE=client_credentials`, or a refresh-token/password grant.
