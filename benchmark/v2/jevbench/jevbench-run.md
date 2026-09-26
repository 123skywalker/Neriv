# JevBench public run

- 数据：`original.jsonl`、`easy.jsonl`、`hard.jsonl`，共 231 条公开题；不含 sealed 题。
- JevBench revision：`1bcc55eb6c8cffde2306b3db03ede39b61c6152a`；适配器：`typesafe`，无 Bearer token。
- Neriv checkpoint：`checkpoint/calibrated-v2`（`20260925T112728Z-s00001000-calibrated`）；`model.pt` SHA-256：`8af9b489d6986b11f6ea7b3390679ed414c909f2133610ebe105e5b0c42bae09`。
- 推理：`Qwen3-0.6B-Instruct`、reference Engine、SDPA、RTX 4060 Laptop GPU；Windows 服务经 WSL 网卡供 JevBench CLI 调用。
- 结果：231/231 完成，231/231 概率合法，137/231 正确（59.31%）；ECE 0.2401，Brier 0.6159；端到端 p50 1.676 秒、p95 7.695 秒。
- `jevbench-public-summary.json` 是 JevBench 汇总；`jevbench-public-results.jsonl` 是逐题结果；`jevbench-raw/` 保存原始请求和响应；`jevbench-manifest.json` 记录数据哈希与运行参数。

本地计算没有公开计费价格。ledger 中的 `$4.62` 是 JevBench 的预算占位额，不是实际支付或推理成本；本次公开集结果不能当作官方含 sealed 题的榜单分数。
