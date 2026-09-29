# Neriv

Neriv 是基于 Qwen3-0.6B 的非生成式概率决策系统：给定背景、问题与 2～16 个候选，直接返回各候选及独立 Reject 槽的概率，不生成文本答案。

![Neriv 控制台演示](demo.gif)

## 组成

- **Model**：Qwen3 编码状态、问题与候选；无候选位置编码的 Set Attention + Pointer Head 对动态候选集合打分。支持 CHOICE、SCORE、NOUL、共享状态与温度校准。
- **Engine**：Paged KV、跨请求 State 前缀缓存、动态批处理、容量背压与超时处理。Linux 可选 FlashInfer；Windows/Linux 均可使用 PyTorch Reference 后端。
- **Agent**：插件式 Decision/Capability、Session 状态、权限策略、审批与审计。提供 `/api/v1/*`、`/v1/systemone`、MCP Server、Python SDK 和 `/ui` 看板。无令牌只能 EVALUATE。

本仓库只保留运行代码、Web UI 与对应测试；不包含训练数据、训练脚本、模型权重或密钥。

## 公开集结果

当前发布权重 `expansion-calibrated-e05` 在 JevBench public original/easy/hard 共 231 题上，使用 Reference Engine 与原生 `typesafe` 适配器：**147/231 正确（63.64%）**，231/231 响应有效，ECE **0.0891**，Brier **0.4852**。配置、逐题结果和原始响应见 [benchmark/v2/jevbench-expansion-calibrated-e05](benchmark/v2/jevbench-expansion-calibrated-e05/README.md)。这是公开集测试，不是包含 sealed 题的官方总榜分；JevBench 未用于本权重的训练或校准。

## 快速启动

权重托管在 [linyaocai/Neriv-v2](https://huggingface.co/linyaocai/Neriv-v2)：`base/` 为 Qwen3 原始权重，`checkpoint/expansion-calibrated-e05/` 为当前校准权重。推荐 Python 3.10+；以下命令适用于 Linux/macOS，Windows 将虚拟环境激活命令换为 `.venv\Scripts\activate`。

```bash
git clone https://github.com/123skywalker/Neriv.git
cd Neriv
python -m venv .venv
source .venv/bin/activate
pip install -e '.[server]'
python -c "from huggingface_hub import snapshot_download; snapshot_download('linyaocai/Neriv-v2', local_dir='weights')"
neriv-serve --checkpoint weights/checkpoint/expansion-calibrated-e05 --model weights/base \
  --engine-backend reference --attention-backend sdpa --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/ui/>。Linux CUDA 环境如已安装 FlashInfer，可改用 `--engine-backend flashinfer`。模型原始权重及许可证见 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。

## 开发验证

```bash
pip install -e '.[dev,server]'
python -m pytest -q
cd web && npm ci && npm run build
```

Web 源码在 `web/`，构建产物写入 `jev_like/agent/static/`，由 FastAPI 挂载到 `/ui`。

## 许可证

Neriv 自研代码按 [Apache License 2.0](LICENSE) 发布；此授权不代表重新授权第三方作品。Neriv 自研的 LoRA、决策头与校准参数在 [Hugging Face 模型仓库](https://huggingface.co/linyaocai/Neriv-v2) 按 Apache-2.0 发布。Qwen3-0.6B 原始权重与 tokenizer 仍由上游按其原始许可证授权。
