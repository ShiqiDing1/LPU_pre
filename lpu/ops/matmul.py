"""Part 1: a tiled matmul assembled only from Python ISA operations."""

from isa import ISA
from utils import *


def op_matmul(left, right, bias):
    """Compute ``left @ right.T + bias`` with L0 tiles.

    The implementation intentionally uses LPU layouts and ISA operations rather
    than delegating the calculation to ``torch.matmul``.
    """
    m, k = left.shape
    n = right.shape[0]
    assert k == right.shape[1] and k % K0 == 0
    assert bias.shape == (n,)

    k1 = ceil_div(k, K0)
    n1 = ceil_div(n, N0)
    tile_m, tile_n, tile_k1 = 32, 24, 2
    isa = ISA()
    left_l2 = mk_to_k1mk0(left)
    right_l2 = mk_to_k1mk0(right)
    result_l2 = torch.zeros((n1, m, N0), dtype=left.dtype)

    for n_start in range(0, n, tile_n):
        n_size = min(tile_n, n - n_start)
        pmb = isa.gdma_gm2pmb(bias, n, n_start, n_size)
        for m_start in range(0, m, tile_m):
            m_size = min(tile_m, m - m_start)
            psb = torch.zeros(
                (ceil_div(m_size, M0), ceil_div(n_size, N0), M0, N0),
                dtype=torch.float32,
            )
            for k1_start in range(0, k1, tile_k1):
                k1_size = min(tile_k1, k1 - k1_start)
                lmb = isa.gdma_gm2lmb(
                    left_l2, m, m_start, k1, k1_start, m_size, k1_size
                )
                rmb = isa.gdma_gm2rmb(
                    right_l2, n, n_start, k1, k1_start, n_size, k1_size
                )
                psb = isa.mxu_matmul(
                    lmb, rmb, pmb, psb, m_size, n_size, k1_size,
                    bias_en=(k1_start == 0),
                    psum_en=(k1_start > 0),
                )

            result_l2 = isa.aru_psb2gm(
                psb, result_l2, None, 1.0, -np.inf, np.inf,
                slice_m=m_size, slice_n=n_size,
                tensor_m=m, tensor_n1=n1,
                start_tensor_m=m_start, start_tensor_n1=n_start // N0,
                arb_en=False, br_m=False, br_n=False, scalar_en=False,
                add_en=False, sub_en=False, max_en=False, min_en=False,
                mul_en=False, div_en=False, neg_en=False, clamp_en=False,
                exp_en=False, sqrt_en=False, pow_en=False, recp_en=False,
            )

    return k1mk0_to_mk(result_l2, n)


# Keep the historical tutorial name as a small compatibility alias.
op_matmul_tile_once = op_matmul
