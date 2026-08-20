# RedditAPIs.com 数据获取与测试报告

## 获取范围

- 主题：Australia–Tuvalu Falepili Union、气候迁移、流动路径、永久居留与相关条约讨论。
- 查询模式：RedditAPIs.com 搜索接口，严格相关性过滤。
- 最终帖子：14 条；最终评论：226 条。
- 核心样本帖子：3 条。
- 帖子发布时间：2023-11-10T04:20:46Z 至 2025-07-26T22:15:25Z。
- 涉及 subreddit：11 个（EcoNewsNetwork, NoFilterNews, Oceania, anime_titties, australia, environment, europeanunion, news, sdrawkcabtidder, veritynews, worldnews）。
- 原始搜索响应：40 条 JSONL 记录，文件 19.5 MB。
- 原始评论响应：81 条 JSONL 记录，文件 6.3 MB。

## 具体字段

- 帖子：ID、subreddit、标题、正文、URL/permalink、作者、发布时间、得分、点赞比例、评论数、NSFW/置顶/锁定状态、查询词、抓取时间、相关性。
- 评论：ID、所属帖子、父评论、正文、作者、发布时间、得分、是否发帖人、层级、抓取时间、相关性。

## 查询词命中

- `Falepili Union`：11 条
- `Tuvalu`：2 条
- `climate migration`：1 条

## 测试结果

| 测试 | 结果 | 详情 |
|---|---:|---|
| 搜索原始 JSONL 可解析 | 通过 | 40/40 行有效 |
| 评论原始 JSONL 可解析 | 通过 | 81/81 行有效 |
| 帖子主键唯一 | 通过 | 14/14 |
| 评论主键唯一 | 通过 | 226/226 |
| 评论关联帖子有效 | 通过 | 226/226 |
| 帖子必填字段完整 | 通过 | reddit_post_id, title, permalink, created_at, query_source, retrieved_at |
| 可用评论必填字段完整 | 通过 | 225 条可用；1 条删除/不可用 |
| 帖子严格相关性过滤 | 通过 | 帖子仅保留 directly relevant |
| 相关主题内评论完整保留 | 通过 | 评论保留直接相关与语境相关 |

总体：9/9 项通过。

## 在线重跑说明

- `key.txt` 已能被项目正确定位，且密钥内容未写入报告或控制台。
- 2026-08-19 已完成密钥鉴权、全站关键词搜索、核心 subreddit 定向搜索及三条相关帖评论树抓取。
- Python 网络请求受沙箱 DNS 限制时，使用已授权的 curl 传输并将完整 JSON 响应导入相同的归一化流程。