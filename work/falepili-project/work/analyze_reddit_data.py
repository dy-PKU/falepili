from pathlib import Path
import json
import pandas as pd

root = Path('work/extracted/falepili-project')
out = Path('outputs/redditapis-result')
out.mkdir(parents=True, exist_ok=True)

posts = pd.read_parquet(root / 'data/processed/reddit_posts.parquet')
comments = pd.read_parquet(root / 'data/processed/reddit_comments.parquet')
posts.to_csv(out / 'reddit_posts.csv', index=False)
comments.to_csv(out / 'reddit_comments.csv', index=False)

def jsonl_stats(path):
    lines = valid = invalid = 0
    payload_types = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            lines += 1
            try:
                obj = json.loads(line)
                valid += 1
                typ = type(obj.get('payload')).__name__ if isinstance(obj, dict) else type(obj).__name__
                payload_types[typ] = payload_types.get(typ, 0) + 1
            except Exception:
                invalid += 1
    return lines, valid, invalid, payload_types

search_raw = root / 'data/raw/reddit_search_responses.jsonl'
comment_raw = root / 'data/raw/reddit_comment_responses.jsonl'
s_stats = jsonl_stats(search_raw)
c_stats = jsonl_stats(comment_raw)

post_ids_unique = posts['reddit_post_id'].nunique() == len(posts)
comment_ids_unique = comments['reddit_comment_id'].nunique() == len(comments)
comment_fk_ok = comments['reddit_post_id'].isin(posts['reddit_post_id']).all()
post_required = ['reddit_post_id','title','permalink','created_at','query_source','retrieved_at']
comment_required = ['reddit_comment_id','reddit_post_id','body','created_at','retrieved_at']
post_required_ok = all(c in posts.columns and posts[c].notna().all() for c in post_required)
available_comments = comments[comments.get('content_status', pd.Series('available', index=comments.index)).eq('available')]
comment_required_ok = all(c in available_comments.columns and available_comments[c].notna().all() for c in comment_required)

tests = [
    ('搜索原始 JSONL 可解析', s_stats[2] == 0, f'{s_stats[1]}/{s_stats[0]} 行有效'),
    ('评论原始 JSONL 可解析', c_stats[2] == 0, f'{c_stats[1]}/{c_stats[0]} 行有效'),
    ('帖子主键唯一', post_ids_unique, f'{posts.reddit_post_id.nunique()}/{len(posts)}'),
    ('评论主键唯一', comment_ids_unique, f'{comments.reddit_comment_id.nunique()}/{len(comments)}'),
    ('评论关联帖子有效', comment_fk_ok, f'{comments.reddit_post_id.isin(posts.reddit_post_id).sum()}/{len(comments)}'),
    ('帖子必填字段完整', post_required_ok, ', '.join(post_required)),
    ('可用评论必填字段完整', comment_required_ok, f'{len(available_comments)} 条可用；{len(comments)-len(available_comments)} 条删除/不可用'),
    ('帖子严格相关性过滤', posts.relevance_status.eq('directly relevant').all(), '帖子仅保留 directly relevant'),
    ('相关主题内评论完整保留', comments.relevance_status.isin(['directly relevant','contextually relevant']).all(), '评论保留直接相关与语境相关'),
]

subs = posts['subreddit'].value_counts().to_dict()
queries = posts['query_source'].value_counts().to_dict()
report = [
    '# RedditAPIs.com 数据获取与测试报告', '',
    '## 获取范围', '',
    '- 主题：Australia–Tuvalu Falepili Union、气候迁移、流动路径、永久居留与相关条约讨论。',
    '- 查询模式：RedditAPIs.com 搜索接口，严格相关性过滤。',
    f'- 最终帖子：{len(posts)} 条；最终评论：{len(comments)} 条。',
    f'- 核心样本帖子：{posts[posts.subreddit_scope.isin(["australia", "AustralianPolitics"])].shape[0]} 条。',
    f'- 帖子发布时间：{posts.created_at.min()} 至 {posts.created_at.max()}。',
    f'- 涉及 subreddit：{len(subs)} 个（{", ".join(sorted(subs))}）。',
    f'- 原始搜索响应：{s_stats[0]} 条 JSONL 记录，文件 {search_raw.stat().st_size/1024/1024:.1f} MB。',
    f'- 原始评论响应：{c_stats[0]} 条 JSONL 记录，文件 {comment_raw.stat().st_size/1024/1024:.1f} MB。', '',
    '## 具体字段', '',
    '- 帖子：ID、subreddit、标题、正文、URL/permalink、作者、发布时间、得分、点赞比例、评论数、NSFW/置顶/锁定状态、查询词、抓取时间、相关性。',
    '- 评论：ID、所属帖子、父评论、正文、作者、发布时间、得分、是否发帖人、层级、抓取时间、相关性。', '',
    '## 查询词命中', '',
]
report += [f'- `{k}`：{v} 条' for k,v in queries.items()]
report += ['', '## 测试结果', '', '| 测试 | 结果 | 详情 |', '|---|---:|---|']
report += [f'| {name} | {"通过" if ok else "失败"} | {detail} |' for name,ok,detail in tests]
report += ['', f'总体：{sum(ok for _,ok,_ in tests)}/{len(tests)} 项通过。', '',
           '## 在线重跑说明', '',
           '- `key.txt` 已能被项目正确定位，且密钥内容未写入报告或控制台。',
           '- 2026-08-19 已完成密钥鉴权、全站关键词搜索、核心 subreddit 定向搜索及三条相关帖评论树抓取。',
           '- Python 网络请求受沙箱 DNS 限制时，使用已授权的 curl 传输并将完整 JSON 响应导入相同的归一化流程。']
(out / 'README.md').write_text('\n'.join(report), encoding='utf-8')

print(json.dumps({'posts':len(posts),'comments':len(comments),'search_raw':s_stats,'comment_raw':c_stats,'tests_passed':sum(ok for _,ok,_ in tests),'tests_total':len(tests)}, ensure_ascii=False, default=str))
