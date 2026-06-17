#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch NCCL DDP smoke test for GPU acceptance")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        fail(f"PyTorch import failed: {exc}")

    if not torch.cuda.is_available():
        fail("CUDA is not available in PyTorch.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        fail(f"GPU count is insufficient: torch sees {device_count}, WORLD_SIZE={world_size}.")
    if local_rank >= device_count:
        fail(f"LOCAL_RANK={local_rank} but torch sees only {device_count} CUDA devices.")
    if args.matrix_size <= 0 or args.iterations <= 0:
        fail("matrix-size and iterations must be positive.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            fail("BF16 requested but this GPU/PyTorch build does not report BF16 support.")
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.manual_seed(20260617 + rank)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=dtype)
    b = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.time()
    total_iters = args.warmup + args.iterations
    checksum = None
    for i in range(total_iters):
        c = a @ b
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        checksum = c[0, 0].float().item()
        if i == args.warmup - 1:
            torch.cuda.synchronize(device)
            start = time.time()
            torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(
        f"rank={rank} local_rank={local_rank} gpu_index={local_rank} world_size={world_size} "
        f"dtype={args.dtype} matrix_size={args.matrix_size} iterations={args.iterations} "
        f"elapsed_sec={elapsed:.3f} max_memory_allocated_mib={max_mem:.1f} checksum={checksum:.6g}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

