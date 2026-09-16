import os
import subprocess
import sys
from pathlib import Path

import torch

from common import K0, LMB_SIZE, M0, MAX_SETWAIT_ID, N0, RMB_SIZE, UB_SIZE
from utils import ceil_div


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("cmodel_runner.py")

CASE4_SHAPE = (192, 192, 1280)
CASE4_L1_TILE = (96, 96, 256)
CASE4_L0_TILE = (32, 32, 64)


def run_cmodel(info, mode, output, timeout=15):
    env = os.environ.copy()
    env["CMODEL_VERBOSE"] = "0"
    command = [
        sys.executable, str(RUNNER), info["taskbin"], mode,
        str(info["result_gm_addr"]), str(info["result_size"]), str(output),
    ]
    return subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=timeout
    )


def decode_result(raw, info):
    words = torch.frombuffer(bytearray(raw), dtype=torch.uint16)
    values = words.view(torch.bfloat16)
    n1, m, n0 = info["layout_shape"]
    layout = values[: n1 * m * n0].reshape(n1, m, n0)
    return layout.permute(1, 0, 2).reshape(m, n1 * n0)[:, : info["shape"][1]]


def error_summary(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return f"max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}"


def validate_case4():
    m, n, k = CASE4_SHAPE
    m_l1, n_l1, k_l1 = CASE4_L1_TILE
    m_l0, n_l0, k_l0 = CASE4_L0_TILE
    l1_trips = ceil_div(m, m_l1) * ceil_div(n, n_l1) * ceil_div(k, k_l1)

    ub_bytes_per_trip = k_l1 * (m_l1 + n_l1) * 2
    assert l1_trips * ub_bytes_per_trip > UB_SIZE
    assert l1_trips > MAX_SETWAIT_ID

    l0_trips_per_l1 = (
        ceil_div(m_l1, m_l0)
        * ceil_div(n_l1, n_l0)
        * ceil_div(k_l1, k_l0)
    )
    lmb_bytes = ceil_div(m_l0, M0) * (k_l0 // K0) * M0 * K0 * 2
    rmb_bytes = ceil_div(n_l0, N0) * (k_l0 // K0) * N0 * K0 * 2
    assert l1_trips * l0_trips_per_l1 * lmb_bytes > LMB_SIZE
    assert l1_trips * l0_trips_per_l1 * rmb_bytes > RMB_SIZE
    return l1_trips
