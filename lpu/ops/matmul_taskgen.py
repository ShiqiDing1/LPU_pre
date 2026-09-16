"""Part 2: sequential TaskGen address-allocation exercise."""

from pathlib import Path

from task_gen import TaskGen
from utils import *
from ops.tiling import *


def op_matmul_taskgen(left, right, bias, *, l1_tile=None, l0_tile=None):
    M_L2 = left.shape[0]
    N_L2 = right.shape[0]
    K_L2 = left.shape[1]
    N1_L2 = ceil_div(N_L2, N0)
    K1_L2 = ceil_div(K_L2, K0)
    assert(K_L2 % K0 == 0)
    assert(K_L2 == right.shape[1])
    M_L1, N_L1, K_L1, M_L0, N_L0, K_L0 = resolve_matmul_tiling(
        M_L2, N_L2, K_L2, l1_tile, l0_tile
    )
    N1_L1 = ceil_div(N_L1, N0)
    K1_L1 = ceil_div(K_L1, K0)
    M1_L0 = ceil_div(M_L0, M0)
    N1_L0 = ceil_div(N_L0, N0)
    K1_L0 = ceil_div(K_L0, K0)
    psb_tile_bytes = ceil_align(M1_L0 * N1_L0 * M0 * N0 * 4, PSB_ALIGN)

    task_gen = TaskGen("op_matmul_taskgen")
    left_k1mk0_l2 = mk_to_k1mk0(left)
    right_k1nk0_l2 = mk_to_k1mk0(right)
    golden_mn = (left.float() @ right.float().T + bias).to(left.dtype)
    golden_n1mn0_l2 = mk_to_k1mk0(golden_mn)

    esize = bias.element_size()
    byte_len = bias.numel() * esize
    pad_bytes = (BUS_WIDTH - (byte_len % BUS_WIDTH)) % BUS_WIDTH
    assert BUS_WIDTH % esize == 0, "The width of the BUS must be an integer multiple of the element byte size."

    extra_elems = pad_bytes // esize
    bias_flat = bias.reshape(-1).contiguous()
    if extra_elems:
        pad = torch.zeros(extra_elems, dtype=bias.dtype, device=bias.device)
        bias_aligned = torch.cat([bias_flat, pad], dim=0)
    else:
        bias_aligned = bias_flat

    left_vec = left_k1mk0_l2.contiguous().view(-1)   # 维持 k1mk0 顺序，只是拉直
    esize_l  = left_vec.element_size()
    pad_l    = (BUS_WIDTH - (left_vec.numel()*esize_l) % BUS_WIDTH) % BUS_WIDTH
    if pad_l:
        left_vec = torch.cat(
            [left_vec, torch.zeros(pad_l // esize_l, dtype=left_vec.dtype, device=left_vec.device)],
            dim=0
        )
    left_aligned = left_vec  # 用它写 GM & 计算 size/addr

    right_vec = right_k1nk0_l2.contiguous().view(-1)
    esize_r  = right_vec.element_size()
    pad_r    = (BUS_WIDTH - (right_vec.numel()*esize_r) % BUS_WIDTH) % BUS_WIDTH
    if pad_r:
        right_vec = torch.cat(
            [right_vec, torch.zeros(pad_r // esize_r, dtype=right_vec.dtype, device=right_vec.device)],
            dim=0
        )
    right_aligned = right_vec

    bias_gm_addr = 0
    bias_gm_size = ceil_align(bias.numel() * bias.element_size(),BUS_WIDTH)
    left_gm_addr = bias_gm_addr + bias_gm_size
    left_gm_size = ceil_align(left_aligned.numel() * left_aligned.element_size(),BUS_WIDTH)
    right_gm_addr = left_gm_addr + left_gm_size
    right_gm_size = ceil_align(right_aligned.size().numel() * right_aligned.element_size(), BUS_WIDTH)
    result_gm_addr = right_gm_addr + right_gm_size
    result_gm_size = ceil_align(golden_n1mn0_l2.size().numel() * golden_n1mn0_l2.element_size(), BUS_WIDTH)
    total_data_size = bias_gm_size + left_gm_size + right_gm_size + result_gm_size
    task_gen.add_input_data(bias_aligned)
    task_gen.add_input_data(left_aligned)
    task_gen.add_input_data(right_aligned)

    ub_addr = 0
    lmb_addr = 0
    lmb_addr_base = 0
    rmb_addr = 0
    rmb_addr_base = 0
    psb_addr = 0
    pmb_addr = 0
    result_ub_addr = 0

    result_ub_bytes = ceil_align(N1_L1 * M_L1 * N0 * 2, UB_ALIGN) # 先划分出一个大空间给result
    ub_ring_start = result_ub_addr + result_ub_bytes
    assert ub_ring_start <= UB_SIZE, "The UB capacity is insufficient to accommodate the result."
    ub_addr = ub_ring_start

    for l1_n_start_in_l2 in range(0, N_L2, N_L1):
        n_size_l1 = min(N_L1, N_L2 - l1_n_start_in_l2)
        for l1_m_start_in_l2 in range(0, M_L2, M_L1):
            m_size_l1 = min(M_L1, M_L2 - l1_m_start_in_l2)
            psb_idx_max = ceil_div(m_size_l1, M_L0) * ceil_div(n_size_l1, N_L0) - 1
            n1_size_l1 = ceil_div(n_size_l1, N0)
            for l1_k1_start_in_l2 in range(0, K1_L2, K1_L1):
                k1_size_l1 = min(K1_L1, K1_L2 - l1_k1_start_in_l2)

                left_bytes = ceil_align(k1_size_l1 * m_size_l1 * K0 * left.element_size(), UB_ALIGN)
                right_bytes = ceil_align(k1_size_l1 * n_size_l1 * K0 * right.element_size(), UB_ALIGN)
                cap = UB_SIZE - ub_ring_start
                assert left_bytes + right_bytes <= cap, "A single left and right tile exceeds the available loop length of UB."

                if ub_addr + left_bytes > UB_SIZE:
                    left_ub_addr = ub_ring_start
                    ub_addr = ub_ring_start + left_bytes
                else:
                    left_ub_addr = ub_addr
                    ub_addr += left_bytes
                task_gen.gdma_gm2ub(left_gm_addr, left_ub_addr, M_L2, l1_m_start_in_l2, K1_L2, l1_k1_start_in_l2, m_size_l1, k1_size_l1)

                if ub_addr + right_bytes > UB_SIZE:
                    right_ub_addr = ub_ring_start
                    ub_addr = ub_ring_start + right_bytes
                else:
                    right_ub_addr = ub_addr
                    ub_addr += right_bytes
                task_gen.gdma_gm2ub(right_gm_addr, right_ub_addr, N_L2, l1_n_start_in_l2, K1_L2, l1_k1_start_in_l2, n_size_l1, k1_size_l1)

                for l0_n_start_in_l1 in range(0, n_size_l1, N_L0):
                    n_size_l0 = min(N_L0, n_size_l1 - l0_n_start_in_l1)
                    pmb_bytes = ceil_align(ceil_div(n_size_l0, N0) * N0 * bias.element_size(), PMB_ALIGN)
                    assert pmb_bytes <= PMB_SIZE, "A single PMB tile exceeds the PMB capacity."
                    if pmb_addr + pmb_bytes > PMB_SIZE:
                        pmb_addr_base = 0
                    else:
                        pmb_addr_base = pmb_addr
                    task_gen.gdma_gm2pmb(bias_gm_addr, pmb_addr_base, N_L2, l0_n_start_in_l1 + l1_n_start_in_l2, n_size_l0)
                    pmb_addr = pmb_addr_base + pmb_bytes

                    for l0_m_start_in_l1 in range(0, m_size_l1, M_L0):
                        m_size_l0 = min(M_L0, m_size_l1 - l0_m_start_in_l1)
                        for l0_k1_start_in_l1 in range(0, k1_size_l1, K1_L0):
                            k1_size_l0 = min(K1_L0, k1_size_l1 - l0_k1_start_in_l1)

                            lmb_bytes = ceil_align(ceil_div(m_size_l0, M0) * k1_size_l0 * M0 * K0 * left.element_size(), LMB_ALIGN)
                            assert lmb_bytes <= LMB_SIZE, "A single LMB tile exceeds the LMB capacity."
                            lmb_addr_base = 0 if lmb_addr + lmb_bytes > LMB_SIZE else lmb_addr

                            rmb_bytes = ceil_align(ceil_div(n_size_l0, N0) * k1_size_l0 * N0 * K0 * right.element_size(), RMB_ALIGN)
                            assert rmb_bytes <= RMB_SIZE, "A single RMB tile exceeds the RMB capacity."
                            rmb_addr_base = 0 if rmb_addr + rmb_bytes > RMB_SIZE else rmb_addr

                            task_gen.ldma_ub2lmb(left_ub_addr, lmb_addr_base, m_size_l1, l0_m_start_in_l1, k1_size_l1, l0_k1_start_in_l1, m_size_l0, k1_size_l0)
                            lmb_addr = lmb_addr_base + lmb_bytes

                            task_gen.ldma_ub2rmb(right_ub_addr, rmb_addr_base, n_size_l1, l0_n_start_in_l1, k1_size_l1, l0_k1_start_in_l1, n_size_l0, k1_size_l0)
                            rmb_addr = rmb_addr_base + rmb_bytes

                            bias_en = (l0_k1_start_in_l1==0 and l1_k1_start_in_l2==0)
                            psum_en = l0_k1_start_in_l1 > 0 or l1_k1_start_in_l2 > 0

                            output_en = (l0_k1_start_in_l1 + k1_size_l0 >= k1_size_l1) and (l1_k1_start_in_l2 + k1_size_l1 >= K1_L2)
                            psb_idx = (l0_m_start_in_l1//M_L0 * ceil_div(n_size_l1, N_L0) + l0_n_start_in_l1//N_L0)
                            if psum_en:
                                task_gen.fence(2)
                            psb_addr = psb_idx * psb_tile_bytes
                            psb_addr_max = psb_idx_max * psb_tile_bytes
                            task_gen.mxu_matmul(lmb_addr=lmb_addr_base, rmb_addr=rmb_addr_base, pmb_addr=pmb_addr_base, psb_addr=psb_addr, slice_m=m_size_l0, slice_n=n_size_l0, slice_k1=k1_size_l0, bias_en=bias_en, psum_en=psum_en)

                            if output_en:
                                l0_n1_start_in_l1 = l0_n_start_in_l1 // N0
                                task_gen.aru_psb2ub(psb_addr=psb_addr, ub_addr=result_ub_addr, arb_addr=0, scalar=1., clamp_min=-np.inf, clamp_max=np.inf,
                                                    slice_m = m_size_l0, slice_n = n_size_l0, tile_m = m_size_l1, tile_n1=n1_size_l1, start_tile_m=l0_m_start_in_l1, start_tile_n1=l0_n1_start_in_l1,
                                                    arb_en=False, br_m=False, br_n=False, scalar_en=False,
                                                    add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False,
                                                    neg_en=False, clamp_en=False, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False,
                                                    reduce_m_en=False, reduce_n_en=False, reduce_mode=0,
                                                    ub_wr_en=True, arb_wr_en=False)

            task_gen.fence(3)
            l1_n1_start_in_l2 = l1_n_start_in_l2 // N0
            task_gen.aru_ub2gm(ub_addr = result_ub_addr, gm_addr = result_gm_addr, arb_addr = 0, scalar=1., clamp_min=-np.inf, clamp_max=np.inf,
                                            tile_m=m_size_l1, tile_n=n_size_l1, tensor_m=M_L2, tensor_n1=N1_L2, start_tensor_m=l1_m_start_in_l2, start_tensor_n1=l1_n1_start_in_l2,
                                            arb_en=False, br_m=False, br_n=False, scalar_en=False,
                                            add_en=False, sub_en=False, max_en=False, min_en=False, mul_en=False, div_en=False,
                                            neg_en=False, clamp_en=False, exp_en=False, sqrt_en=False, pow_en=False, recp_en=False)
    golden_n1mn0_l2_aligned = golden_n1mn0_l2.contiguous().view(-1)
    esize = golden_n1mn0_l2_aligned.element_size()
    assert BUS_WIDTH % esize == 0, "The width of the BUS must be an integer multiple of the element byte size."
    pad_bytes = (BUS_WIDTH - (golden_n1mn0_l2_aligned.numel() * esize) % BUS_WIDTH) % BUS_WIDTH
    if pad_bytes:
        extra = pad_bytes // esize
        golden_n1mn0_l2_aligned = torch.cat([golden_n1mn0_l2_aligned, torch.zeros(extra, dtype=golden_n1mn0_l2_aligned.dtype, device=golden_n1mn0_l2_aligned.device)], dim=0)
    task_gen.add_output_data(golden_n1mn0_l2_aligned)
    task_gen.generate()
    return {
        "taskbin": str(Path("op_matmul_taskgen.bin").resolve()),
        "result_gm_addr": result_gm_addr,
        "result_size": result_gm_size,
        "shape": (M_L2, N_L2),
        "layout_shape": tuple(golden_n1mn0_l2.shape),
    }
