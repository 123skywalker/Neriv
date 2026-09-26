# 运行环境与算子选择

## 本机结论

本机是 RTX 4060 Laptop 8 GB。训练与正式高性能推理使用 WSL2/Linux；Windows 只保留 PyTorch SDPA Reference Backend，便于开发、兼容推理和数值对拍。

已验证的 WSL 环境：

```text
/root/miniconda3/bin/python
PyTorch 2.6.0+cu124
CUDA 可用，BF16 可用
Triton 3.2.0
torch.compile / Inductor 可用
Flash SDPA 可用
```

Windows 原生环境虽然能运行 `flash_attn` 前向，但没有 Triton，GPU `torch.compile` 不可用，且该组合的 LoRA 反向出现过非有限参数。训练脚本因此优先采用 Linux BF16 + PyTorch fused SDPA；`--compile` 只编译形状较稳定的 Pointer 子图，短任务通常不值得承担冷启动开销。

## WSL 文件布局

模型和训练输出放 WSL ext4，项目源码可以继续位于 Windows 工作区：

```bash
mkdir -p /root/jev-like-assets/models /root/jev-like-runs
cp -a /mnt/d/Tech_Test/miniVllm/models/Qwen3-0.6B-Instruct \
  /root/jev-like-assets/models/
cd /mnt/d/Tech_Test/Jev_Like
/root/miniconda3/bin/python scripts/verify_linux_runtime.py
```

不要在 WSL 内安装 Linux NVIDIA 显示驱动；WSL CUDA 由 Windows NVIDIA 驱动映射。

## FlashInfer 正式 Engine

当前 PyTorch/CUDA 组合使用匹配的 Linux wheel：

```bash
/root/miniconda3/bin/python -m pip install \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl'
```

Neriv 的 `FlashInferPagedBackend` 直接使用物理 Paged KV tensor、page table、`append_paged_kv_cache` 与 Batch Paged Prefill。`auto` 仅在 Linux CUDA 上选择 FlashInfer；未安装或初始化失败会直接报错，不会静默把正式后端降级成 Reference。

Reference Backend 在 Windows/Linux 均可运行，使用同一套 page 生命周期和 PyTorch SDPA，作用是兼容推理与 FlashInfer 数值基准，不作为 Linux 正式发布后端。

## 训练阶段与恢复语义

```bash
# Head warm-up
/root/miniconda3/bin/python -m jev_like.training.trainer \
  --model /root/jev-like-assets/models/Qwen3-0.6B-Instruct \
  --data data/normalized/head_train.jsonl \
  --output /root/jev-like-runs/head --stage head

# 合法的跨阶段继承：模型权重继承，新优化器、step 0
/root/miniconda3/bin/python -m jev_like.training.trainer \
  --model /root/jev-like-assets/models/Qwen3-0.6B-Instruct \
  --data data/normalized/general_train.jsonl \
  --output /root/jev-like-runs/general --stage general-sft \
  --parent-checkpoint /root/jev-like-runs/head/checkpoints/<version>

# 同阶段精确恢复：模型、优化器、scheduler、RNG、epoch/batch 游标全部恢复
/root/miniconda3/bin/python -m jev_like.training.trainer \
  --model /root/jev-like-assets/models/Qwen3-0.6B-Instruct \
  --data data/normalized/general_train.jsonl \
  --output /root/jev-like-runs/general --stage general-sft --resume latest
```

每个 checkpoint 都含 `checkpoint_manifest.json`，固化 Contract、backbone/tokenizer revision、dataset manifest hash、父 checkpoint hash、训练阶段、seed 和代码 revision。
