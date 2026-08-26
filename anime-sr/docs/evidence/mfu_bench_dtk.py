#!/usr/bin/env python
# mfu_bench.py — RFMSR-0.3B (DiT-style) 吞吐/MFU 基准（DTK/HIP 双卡，逐卡独立进程）
# 用法（每张卡各起一个进程）：
#   CUDA_VISIBLE_DEVICES=0 /sakuramoon-runtime/sakuramoon-dtk-venv/bin/python mfu_bench.py --out /root/mfu_gpu0.json
#   CUDA_VISIBLE_DEVICES=1 /sakuramoon-runtime/sakuramoon-dtk-venv/bin/python mfu_bench.py --out /root/mfu_gpu1.json
# 模型：dim=1024, depth=24, heads=16, FFN=4x, seq=1024 tokens, in_dim=128 (Mage-VAE latent ch)，bf16
# FLOPs 解析：fwd 每样本 = 24*(8d^2 + 4n^2 d) + proj；训练 iter ≈ 3x fwd (fwd + 2x bwd)
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.heads = heads
        self.hdim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.hdim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (b, heads, n, hdim)
        o = torch.matmul(torch.matmul(q, k.transpose(-2, -1)) / (self.hdim ** 0.5), v)
        x = x + self.proj(o.transpose(1, 2).reshape(b, n, d))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class DiT03(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, in_dim=128, use_ckpt=False):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim, bias=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.out_proj = nn.Linear(dim, in_dim, bias=False)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks:
            if self.use_ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.out_proj(x)


def fwd_flops_per_sample(dim, seq):
    # 每 block: QKV+O 线性 8d^2 (每 d^2 计 2 FLOP × 4 个 d×d 矩阵... 解析: 4 个 d->d 线性 = 8d^2)
    # attention: QK^T 2n^2d + AV 2n^2d = 4n^2d ; FFN: up 2*d*4d + down 2*4d*d = 16d^2
    # 合计/block = 24d^2 + 4n^2d
    per_block = 24 * dim * dim + 4 * seq * seq * dim
    proj = 2 * (2 * 128 * dim + 2 * dim * 128)  # in/out proj
    return depth_blocks * per_block + proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="结果 JSON 输出路径")
    ap.add_argument("--batches", default="8,16,32,64", help="batch ladder，逗号分隔")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--peak-tflops", type=float, default=50.0,
                    help="设备 bf16 dense 标称峰值 (TFLOPS)，用于折算 MFU；按实际 DCU 规格调整")
    ap.add_argument("--checkpoint", action="store_true", help="梯度检查点（省激活显存，重计算开销 ~10%）")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "cuda(HIP) 不可用"
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    global depth_blocks
    dim, depth_blocks, heads, in_dim = 1024, 24, 16, 128
    fwd_ps = fwd_flops_per_sample(dim, a.seq)

    model = DiT03(dim, depth_blocks, heads, in_dim, use_ckpt=a.checkpoint).to(dev)
    model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    results = []
    for batch in [int(x) for x in a.batches.split(",")]:
        x = torch.randn(batch, a.seq, in_dim, device=dev, dtype=torch.bfloat16)
        target = torch.randn_like(x)

        def step():
            opt.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opt.step()
            return loss

        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ips = a.iters / dt
        flops_iter = batch * fwd_ps * 3  # fwd + 2x bwd
        gbs = flops_iter * ips / 1e9
        mfu = gbs * 1e9 / (a.peak_tflops * 1e12)
        results.append(dict(batch=batch, iters_per_s=round(ips, 3),
                            giga_flops_per_s=round(gbs, 1),
                            mfu_pct_vs_peak=round(100 * mfu, 1)))
        print(f"batch={batch:>3}  {ips:8.3f} it/s  {gbs:9.1f} GFLOPs/s  "
              f"MFU={100 * mfu:6.2f}% (vs {a.peak_tflops:.0f} TFLOPS)")

    out = dict(
        device=props.name,
        mem_gb=round(props.total_memory / 2 ** 30, 1),
        params_M=round(n_params / 1e6, 1),
        dtype="bf16", seq=a.seq, iters=a.iters, peak_tflops=a.peak_tflops,
        results=results,
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
