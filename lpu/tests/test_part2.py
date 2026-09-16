#!/usr/bin/env python3
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_test import (
    CASE4_L0_TILE,
    CASE4_L1_TILE,
    CASE4_SHAPE,
    decode_result,
    error_summary,
    run_cmodel,
    validate_case4,
)
from ops.matmul_taskgen import op_matmul_taskgen


def run_case(name, shape, seed, l1_tile=None, l0_tile=None, timeout=15):
    m, n, k = shape
    np.random.seed(seed)
    torch.manual_seed(seed)
    left = torch.randn(m, k, dtype=torch.bfloat16)
    right = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.float32)
    golden = (left.float() @ right.float().T + bias).to(torch.bfloat16)

    with tempfile.TemporaryDirectory(prefix=f"lpu_part2_{name}_") as tmp:
        old = os.getcwd()
        os.chdir(tmp)
        try:
            info = op_matmul_taskgen(
                left, right, bias, l1_tile=l1_tile, l0_tile=l0_tile
            )
        finally:
            os.chdir(old)
        output = Path(tmp) / "sequential.bin"
        proc = run_cmodel(info, "sequential", output, timeout=timeout)
        if proc.returncode != 0:
            print(proc.stderr.strip() or proc.stdout.strip())
            raise RuntimeError(f"cmodel sequential exited {proc.returncode}")
        actual = decode_result(output.read_bytes(), info)
        passed = torch.allclose(actual.float(), golden.float(), atol=0.35, rtol=0.08)
        print(f"  {name} sequential vs PyTorch:", error_summary(actual, golden))
        return passed


def main():
    results = []
    try:
        results.append(run_case("case-1", (19, 17, 32), 2202))
        results.append(
            run_case(
                "case-2",
                (64, 64, 512),
                2203,
                (64, 64, 128),
                (32, 32, 64),
            )
        )
        results.append(
            run_case(
                "case-3",
                (64, 128, 128),
                2204,
                (64, 64, 128),
                (32, 32, 64),
            )
        )
        validate_case4()
        results.append(
            run_case(
                "case-4",
                CASE4_SHAPE,
                2205,
                CASE4_L1_TILE,
                CASE4_L0_TILE,
                timeout=60,
            )
        )
    except Exception as exc:
        print(f"  Part 2 execution failed: {type(exc).__name__}: {exc}")
        results.append(False)

    passed = all(results)
    print("[Part2] PASS" if passed else "[Part2] FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
