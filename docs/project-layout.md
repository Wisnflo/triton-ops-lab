# 项目整理与分享

| 位置 | 用途 | 上传 GitHub |
|---|---|---|
| `src/`、`tests/`、`configs/` | 实现、验证、固定实验配置 | 是 |
| `scripts/`、`notebooks/` | 本地和 GPU 执行入口；notebook 不带输出 | 是 |
| `docs/`、`README.md` | 方法、协议、历史摘要、后续安排 | 脱敏后上传 |
| `tools/export_repository.py` | 构建并检查分享快照 | 是 |
| `local-data/downloads/` | 用户要求收回项目内的浏览器下载副本和私有清单 | 否 |
| `local-data/publish/` | 隔离的分享工作树、隐私检查和上传记录 | 否 |
| 工作区 `storage/runs/triton/` | 已验收的原始证据；保留原位置以维持审计路径 | 否 |
| `.venv/`、缓存与原 `.git/` | 独立运行环境和本地审计历史 | 否 |

新增运行产物仍按原约定写入工作区存储。`local-data/downloads/` 是此次归档下载副本的安排，不是新的 benchmark 输出根目录。源码和锁文件不随归档移动而变更。

分享流程在本项目使用 `uv run --frozen python tools/export_repository.py --output local-data/publish/<new-name>`。它仅收集已纳入 Git 的源码文件，拒绝凭据、私人地址、带输出 notebook、原始数据与不在清单内的路径。文档中的本机位置、私人 Colab 地址和账号用量会在分享副本中脱敏；本地原文及验收历史保持可追溯。

分享目录中的 `repository/` 应单独建立干净 Git 历史，以 GitHub 的 noreply 邮箱提交；不要把本地原有历史、全部分支或标签推送到分享仓库。后续更新重新生成并检查快照，再按正常提交更新远程，避免覆盖他人修改。原工作仓库不自动配置推送远程。
