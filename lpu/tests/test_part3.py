#!/usr/bin/env python3
import os
import subprocess
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
from ops.matmul_sync import op_matmul_sync


def run_case(name, shape, seed, l1_tile, l0_tile, timeout):
    np.random.seed(seed)
    torch.manual_seed(seed)
    m, n, k = shape
    left = torch.randn(m, k, dtype=torch.bfloat16)
    right = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.float32)
    golden = (left.float() @ right.float().T + bias).to(torch.bfloat16)

    with tempfile.TemporaryDirectory(prefix=f"lpu_part3_{name}_") as tmp:
        old = os.getcwd()
        os.chdir(tmp)
        try:
            info = op_matmul_sync(
                left,
                right,
                bias,
                l1_tile=l1_tile,
                l0_tile=l0_tile,
            )
        finally:
            os.chdir(old)
        seq_path = Path(tmp) / "sequential.bin"
        par_path = Path(tmp) / "parallel.bin"
        seq = run_cmodel(info, "sequential", seq_path, timeout=timeout)
        if seq.returncode != 0:
            raise RuntimeError(f"sequential cmodel exited {seq.returncode}")
        try:
            par = run_cmodel(info, "parallel", par_path, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("parallel cmodel timed out")
        if par.returncode != 0:
            raise RuntimeError(f"parallel cmodel exited {par.returncode}")
        sequential = decode_result(seq_path.read_bytes(), info)
        parallel = decode_result(par_path.read_bytes(), info)
        seq_ok = torch.allclose(sequential.float(), golden.float(), atol=0.35, rtol=0.08)
        par_ok = torch.allclose(parallel.float(), sequential.float(), atol=0.02, rtol=0.01)
        print(f"  {name} sequential:", error_summary(sequential, golden))
        print(f"  {name} parallel:", error_summary(parallel, sequential))
        return seq_ok and par_ok


def main():
    validate_case4()
    cases = [
        ("case-1", (32, 32, 32), 3301, (32, 32, 32), (16, 8, 16), 8),
        ("case-2", (32, 32, 544), 3302, (32, 32, 32), (32, 32, 32), 8),
        ("case-3", (128, 128, 256), 3303, (128, 128, 256), (32, 32, 64), 15),
        ("case-4", CASE4_SHAPE, 3304, CASE4_L1_TILE, CASE4_L0_TILE, 60),
    ]
    passed = True
    for args in cases:
        try:
            if not run_case(*args):
                passed = False
                break
        except Exception as exc:
            print(f"  {args[0]} failed: {type(exc).__name__}: {exc}")
            passed = False
            break

    print("[Part3] PASS" if passed else "[Part3] FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
