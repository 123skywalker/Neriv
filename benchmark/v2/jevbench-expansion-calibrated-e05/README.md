# JevBench public · expansion-calibrated-e05

- 日期：2026-09-28；checkpoint：`checkpoint/expansion-calibrated-e05`（`model.pt` SHA-256：`426eeb91fc6116c11d895cc9a927b0b9f8c6e7a54186af6cc0657e98bbd5c4a5`）。
- JevBench 源码固定到 `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`；任务为该版本的 `datasets/public/{original,easy,hard}.jsonl`，本地副本位于 `data/expansion/raw/jevbench_public/`。合并任务 hash：`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`。
- 本机 WSL、RTX 4060 Laptop 8 GiB；`neriv-serve --engine-backend reference --attention-backend sdpa`；`typesafe` adapter，无 Bearer token，逐题顺序请求。

| 指标 | 结果 |
| --- | ---: |
| 有效 / 总题数 | 231 / 231 |
| 正确题数 / Accuracy | 147 / 63.64% |
| Macro Accuracy | 59.45% |
| ECE | 0.0891 |
| Brier Mean | 0.4852 |
| Ordinal MAE | 0.5727 |
| 延迟 p50 / p95 | 1.62 s / 7.07 s |

`results.jsonl` 是逐题结果，`public-summary.json` 是 JevBench 汇总，`manifest.json` 记录运行配置，`raw/` 保存原始响应。`ledger.jsonl` 的 4.62 美元是每题 0.02 美元的预留记账，不是本机推理实际费用。这里只测公开题，不是包含 sealed 题、定价与正式速度条件的官方总榜分。
