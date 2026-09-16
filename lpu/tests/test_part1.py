#!/usr/bin/env python3
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isa import ISA
from ops.matmul import op_matmul
from common import K0, M0, N0
from utils import ceil_div, mk_to_k1mk0


def check(name, fn, golden, atol=0.08, rtol=0.05):
    try:
        actual = fn()
        passed = torch.allclose(actual.float(), golden.float(), atol=atol, rtol=rtol)
        diff = (actual.float() - golden.float()).abs()
        detail = f"max_abs={diff.max().item():.6g}"
    except Exception as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"  {name}: {'ok' if passed else 'failed'} ({detail})")
    return passed


def main():
    torch.manual_seed(2026)
    results = []

    gm = mk_to_k1mk0(torch.randn(37, 32, dtype=torch.bfloat16))
    padded = torch.zeros((ceil_div(19, M0) * M0, 4, K0), dtype=gm.dtype)
    padded[:19] = gm[:, 5:24].permute(1, 0, 2)
    golden_lmb = padded.reshape(ceil_div(19, M0), M0, 4, K0).permute(0, 2, 1, 3)
    results.append(check(
        "case 1",
        lambda: ISA().gdma_gm2lmb(gm, 37, 5, 4, 0, 19, 4),
        golden_lmb,
        atol=0, rtol=0,
    ))

    gm = mk_to_k1mk0(torch.randn(29, 40, dtype=torch.bfloat16))
    padded = torch.zeros((ceil_div(17, N0) * N0, 3, K0), dtype=gm.dtype)
    padded[:17] = gm[1:4, 7:24].permute(1, 0, 2)
    golden_rmb = padded.reshape(ceil_div(17, N0), N0, 3, K0).permute(0, 2, 1, 3)
    results.append(check(
        "case 2",
        lambda: ISA().gdma_gm2rmb(gm, 29, 7, 5, 1, 17, 3),
        golden_rmb,
        atol=0, rtol=0,
    ))

    left = torch.randn(47, 16, dtype=torch.bfloat16)
    right = torch.randn(41, 16, dtype=torch.bfloat16)
    bias = torch.randn(41, dtype=torch.float32)
    golden = (left.float() @ right.float().T + bias).to(torch.bfloat16)
    results.append(check("case 3", lambda: op_matmul(left, right, bias), golden, 0.35, 0.08))

    left = torch.randn(23, 64, dtype=torch.bfloat16)
    right = torch.randn(19, 64, dtype=torch.bfloat16)
    bias = torch.randn(19, dtype=torch.float32)
    golden = (left.float() @ right.float().T + bias).to(torch.bfloat16)
    results.append(check("case 4", lambda: op_matmul(left, right, bias), golden, 0.35, 0.08))

    if all(results):
        print("[Part1] PASS")
        return 0
    print("[Part1] FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
