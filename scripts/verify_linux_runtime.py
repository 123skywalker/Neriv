"""验证 Linux CUDA、Triton、Inductor 与 Flash SDPA 运行时。"""

import json

import torch


def main() -> None:
    """运行最小 GPU 编译计算并输出机器可读结果。"""

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    try:
        import triton
    except ImportError as error:
        raise SystemExit("Triton 不可用") from error
    compiled = torch.compile(lambda value: torch.sin(value) * 2, backend="inductor")
    value = torch.randn(4096, device="cuda")
    correct = torch.allclose(compiled(value), torch.sin(value) * 2)
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "triton": triton.__version__,
        "inductor": bool(correct),
        "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

