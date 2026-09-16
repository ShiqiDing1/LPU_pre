from utils import *
from common import *
from semantic import *

class ISA:
    def __init__(self, name = "", version = ""):
        self.name = name
        self.version = version
        self.set_wait_table = np.zeros((4,4,16), dtype=np.int16)

    def __str__(self):
        return f"ISA: {self.name}, Version: {self.version}"

    def gdma_gm2ub(self, gm_k1mk0, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, tile_m, tile_k1, dsize=2):
        assert gm_k1mk0.shape[0] == tensor_k1
        assert gm_k1mk0.shape[1] == tensor_m
        K0 = K0_Byte // dsize
        ub_k1mk0 = torch.zeros((tile_k1, tile_m, K0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for k1 in range(tile_k1):
            for m in range(tile_m):
                if (start_tensor_k1 + k1) < tensor_k1 and (start_tensor_m + m) < tensor_m:
                    ub_k1mk0[k1, m] = gm_k1mk0[start_tensor_k1 + k1, start_tensor_m + m]
        return ub_k1mk0

    def gdma_gm2lmb(self, gm_k1mk0, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, slice_m, slice_k1, dsize=2):
        assert gm_k1mk0.shape[0] == tensor_k1
        assert gm_k1mk0.shape[1] == tensor_m
        K0 = K0_Byte // dsize
        M1 = ceil_div(slice_m, M0)
        lmb_m1k1m0k0 = torch.zeros((M1, slice_k1, M0, K0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for m1 in range(M1):
            for k1 in range(slice_k1):
                for m0 in range(M0):
                    if (m1 * M0 + m0) < slice_m and (start_tensor_m + m1 * M0 + m0) < tensor_m and (start_tensor_k1 + k1) < tensor_k1:
                        lmb_m1k1m0k0[m1, k1, m0] = gm_k1mk0[start_tensor_k1 + k1, start_tensor_m + m1 * M0 + m0]
        return lmb_m1k1m0k0

    def gdma_gm2rmb(self, gm_k1nk0, tensor_n, start_tensor_n, tensor_k1, start_tensor_k1, slice_n, slice_k1, dsize=2):
        assert gm_k1nk0.shape[0] == tensor_k1
        assert gm_k1nk0.shape[1] == tensor_n
        K0 = K0_Byte // dsize
        N1 = ceil_div(slice_n, N0)
        rmb_n1k1n0k0 = torch.zeros((N1, slice_k1, N0, K0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for n1 in range(N1):
            for k1 in range(slice_k1):
                for n0 in range(N0):
                    if (n1 * N0 + n0) < slice_n and (start_tensor_n + n1 * N0 + n0) < tensor_n and (start_tensor_k1 + k1) < tensor_k1:
                        rmb_n1k1n0k0[n1, k1, n0] = gm_k1nk0[start_tensor_k1 + k1, start_tensor_n + n1 * N0 + n0]
        return rmb_n1k1n0k0

    def gdma_gm2pmb(self, gm_n, tensor_n, start_tensor_n, slice_n, dsize=2):
        assert gm_n.shape[0] == tensor_n
        N1 = ceil_div(slice_n, N0)
        pmb_n1n0 = torch.zeros((N1, N0), dtype=torch.float32)
        for n1 in range(N1):
            for n0 in range(N0):
                if (n1 * N0 + n0) < slice_n:
                    pmb_n1n0[n1, n0] = gm_n[start_tensor_n + n1 * N0 + n0]
        return pmb_n1n0

    def gdma_scalar2ub(self, tile_m, tile_k1, scalar, dsize=2):
        return torch.full((tile_k1, tile_m, K0), scalar, dtype=torch.bfloat16)

    def ldma_ub2lmb(self, ub_k1mk0, tile_m, start_tile_m, start_tile_k1, slice_m, slice_k1, dsize=2):
        K0 = K0_Byte // dsize
        K1 = slice_k1
        M1 = ceil_div(slice_m, M0)
        assert(ub_k1mk0.shape[1] == tile_m)
        lmb_m1k1m0k0 = torch.zeros((M1, K1, M0, K0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for m1 in range(M1):
            for k1 in range(K1):
                for m0 in range(M0):
                    m = m1 * M0 + m0
                    if(m < slice_m):
                        lmb_m1k1m0k0[m1][k1][m0] = ub_k1mk0[start_tile_k1+k1][start_tile_m+m]
        return lmb_m1k1m0k0

    def ldma_ub2rmb(self, ub_k1mk0, tile_n, start_tile_n, start_tile_k1, slice_n, slice_k1, dsize=2):
        K0 = K0_Byte // dsize
        K1 = slice_k1
        N1 = ceil_div(slice_n, N0)
        assert(ub_k1mk0.shape[1] == tile_n)

        rmb_n1k1n0k0 = torch.zeros((N1, K1, N0, K0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for n1 in range(N1):
            for k1 in range(K1):
                for n0 in range(N0):
                    n = n1 * N0 + n0
                    if(n < slice_n):
                        rmb_n1k1n0k0[n1][k1][n0] = ub_k1mk0[start_tile_k1+k1][start_tile_n+n]
        return rmb_n1k1n0k0

    def ldma_ub2rmb_transpose(self, ub_k1nk0, tile_n, start_tile_n, start_tile_k1, slice_n, slice_k1, dsize=2):
        K = slice_k1 * K0
        K1 = ceil_div(K, N0)
        N1 = ceil_div(slice_n, K0)
        rmb_k1n1k0n0 = torch.zeros((K1, N1, K0, N0), dtype=torch.int8 if dsize == 1 else torch.bfloat16)
        for k1 in range(K1):
            for n1 in range(N1):
                for k0 in range(N0):
                    for n0 in range(K0):
                        n = n1 * K0 + n0
                        if n < slice_n:
                            rmb_k1n1k0n0[k1, n1, k0, n0] = ub_k1nk0[start_tile_k1 + k1, start_tile_n + n, k0]
        return rmb_k1n1k0n0

    def mxu_matmul(self, lmb_m1k1m0k0, rmb_n1k1n0k0, pmb_n1n0, psb_m1n1m0n0, slice_m, slice_n, slice_k1, bias_en, psum_en, dtype='bf16'): #int8->bf16
        """
        dtype: 'int8' or 'bf16'
        """
        assert(lmb_m1k1m0k0.shape[1] == rmb_n1k1n0k0.shape[1])
        assert(lmb_m1k1m0k0.shape[3] == rmb_n1k1n0k0.shape[3])
        M1 = ceil_div(slice_m, M0)
        N1 = ceil_div(slice_n, N0)
        K1 = slice_k1
        if dtype == 'int8':
            K0 = K0_Byte // 1
            lmb_m1k1m0k0 = lmb_m1k1m0k0.to(torch.int32)
            rmb_n1k1n0k0 = rmb_n1k1n0k0.to(torch.int32)
            if(bias_en):
                pmb_n1n0 = pmb_n1n0.to(torch.int32)
        elif dtype == 'bf16':
            K0 = K0_Byte // 2
            lmb_m1k1m0k0 = lmb_m1k1m0k0.to(torch.float32)
            rmb_n1k1n0k0 = rmb_n1k1n0k0.to(torch.float32)
            if(bias_en):
                pmb_n1n0 = pmb_n1n0.to(torch.float32)
        else:
            assert False, f"unsupported mxu_matmul dtype: {dtype}"

        for m1 in range(M1):
            for n1 in range(N1):
                temp = torch.zeros((M0, N0), dtype=torch.int32 if dtype == 'int8' else torch.float32)
                for k1 in range(K1):
                    temp += torch.matmul(lmb_m1k1m0k0[m1][k1], rmb_n1k1n0k0[n1][k1].transpose(0, 1))
                if(bias_en):
                    for n0 in range(N0):
                        temp[:, n0] += pmb_n1n0[n1][n0]
                if psum_en:
                    psb_m1n1m0n0[m1][n1] += temp
                else:
                    psb_m1n1m0n0[m1][n1] = temp
        return psb_m1n1m0n0

    # psb里矩阵的layout是m1n1m0n0, 但是slice_m和slice_n可能不是m0和n0的整数倍, 超出的部分不应该参与reduce, 所以用slice_m和slice_n而不是slice_m1和slice_n1。
    def aru_psb2ub(self, psb_m1n1m0n0, ub_n1mn0, arb, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                  reduce_m_en, reduce_n_en, reduce_mode,
                  ub_wr_en, arb_wr_en, ub_atomic_mode=0):

        assert(ub_n1mn0.shape[0] == tile_n1)
        assert(ub_n1mn0.shape[1] == tile_m)
        assert ub_atomic_mode in (0, 1, 2), "ub_atomic_mode must be 0(no-atomic)/1(add)/2(mul)"
        slice_n1 = ceil_div(slice_n, N0)

        if(arb_en == False and scalar_en == False):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = None
        elif(arb_en == True and scalar_en == False):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = Broadcast(arb, slice_m, slice_n, br_m, br_n).to(torch.float32)
        elif(arb_en == False and scalar_en == True):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = torch.full_like(x1, scalar)
        else:
            assert False, "invalid configuration"

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

        if(reduce_m_en or reduce_n_en):
            assert(arb_wr_en)
            reduce = Reduce(m1k1m0k0_to_mk(x, slice_m, slice_n), reduce_m_en, reduce_n_en, reduce_mode)

        if (ub_wr_en == True):
            x = m1k1m0k0_to_k1mk0(x, slice_m)
            for n1 in range(slice_n1):
                for m in range(slice_m):
                    for n0 in range(N0):
                        n_idx = n1 * N0 + n0
                        m_idx = m + start_tile_m
                        if(m_idx < tile_m):
                            val = x[n1, m, n0] if n_idx < slice_n else 0
                            if ub_atomic_mode == 0:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] = val
                            elif ub_atomic_mode == 1:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] += val
                            else:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] *= val

        if(ub_wr_en == True and arb_wr_en == True):
            return ub_n1mn0, reduce
        elif(ub_wr_en == True and arb_wr_en == False):
            return ub_n1mn0
        elif(ub_wr_en == False and arb_wr_en == True):
            return reduce
        else:
            assert False, "invalid configuration"

    def aru_psb2gm(self, psb_m1n1m0n0, gm_n1mn0, arb, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):

        assert(gm_n1mn0.shape[0] == tensor_n1)
        assert(gm_n1mn0.shape[1] == tensor_m)
        slice_n1 = ceil_div(slice_n, N0)

        if(arb_en == False and scalar_en == False):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = None
        elif(arb_en == True and scalar_en == False):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = Broadcast(arb, slice_m, slice_n, br_m, br_n).to(torch.float32)
        elif(arb_en == False and scalar_en == True):
            x1 = psb_m1n1m0n0.to(torch.bfloat16).to(torch.float32)
            x2 = torch.full_like(x1, scalar)
        else:
            assert False, "invalid configuration"

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

        x = m1k1m0k0_to_k1mk0(x, slice_m)
        for n1 in range(slice_n1):
            for m in range(slice_m):
                for n0 in range(N0):
                    m_idx = m + start_tensor_m
                    if(m_idx < tensor_m):
                        gm_n1mn0[n1 + start_tensor_n1, m_idx, n0] = x[n1, m, n0]

        return gm_n1mn0

    # 从UB到UB, 也可能只是取一块算, 所以还是有slice_m和slice_n域段, 后面可能会调整。
    def aru_ub2ub(self, ub_n1mn0, arb, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                  reduce_m_en, reduce_n_en, reduce_mode,
                  ub_wr_en, arb_wr_en, ub_atomic_mode=0):

        assert(ub_n1mn0.shape[0] == tile_n1)
        assert(ub_n1mn0.shape[1] == tile_m)
        assert ub_atomic_mode in (0, 1, 2), "ub_atomic_mode must be 0(no-atomic)/1(add)/2(mul)"
        slice_n1 = ceil_div(slice_n, N0)
        reduce = 0

        # 如果arb_en为true但是br_n与br_m都为false，代表做的是vector的binary，对ub以及arb都不做broadcast
        # 这种情况下按照k1mk0直接计算，会将arb在broadcast中转为k1mk0
        if(arb_en == False and scalar_en == False):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m]).to(torch.float32)
            x2 = None
        elif(arb_en == True and scalar_en == False):
            if(not br_m and not br_n):
                x1 = ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m].to(torch.float32)
            else:
                x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m]).to(torch.float32)
            x2 = Broadcast(arb, slice_m, slice_n, br_m, br_n).to(torch.float32)
        elif(arb_en == False and scalar_en == True):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m]).to(torch.float32)
            x2 = torch.full_like(x1, scalar)
        else:
            assert False, "invalid configuration"

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)


        if(reduce_m_en or reduce_n_en):
            assert(arb_wr_en)
            if(not br_m and not br_n and arb_en):
                reduce = Reduce(k1mk0_to_mk(x, slice_n), reduce_m_en, reduce_n_en, reduce_mode)
            else:
                reduce = Reduce(m1k1m0k0_to_mk(x, slice_m, slice_n), reduce_m_en, reduce_n_en, reduce_mode)

        if (ub_wr_en == True):
            if not (not br_m and not br_n and arb_en):
                x = m1k1m0k0_to_k1mk0(x, slice_m)
            for n1 in range(slice_n1):
                for m in range(slice_m):
                    for n0 in range(N0):
                        n_idx = n1 * N0 + n0
                        m_idx = m + start_tile_m
                        if(m_idx < tile_m):
                            val = x[n1, m, n0] if n_idx < slice_n else 0
                            if ub_atomic_mode == 0:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] = val
                            elif ub_atomic_mode == 1:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] += val
                            else:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] *= val

        if(ub_wr_en == True and arb_wr_en == True):
            return ub_n1mn0, reduce
        elif(ub_wr_en == True and arb_wr_en == False):
            return ub_n1mn0
        elif(ub_wr_en == False and arb_wr_en == True):
            return reduce
        else:
            assert False, "invalid configuration"

    # 用于element wise算子, 比如x + matmul(x)
    def aru_dual2ub(self, psb_m1n1m0n0, ub_n1mn0, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                  reduce_m_en, reduce_n_en, reduce_mode,
                  ub_wr_en, arb_wr_en, ub_atomic_mode=0):

        assert(ub_n1mn0.shape[0] == tile_n1)
        assert(ub_n1mn0.shape[1] == tile_m)
        assert ub_atomic_mode in (0, 1, 2), "ub_atomic_mode must be 0(no-atomic)/1(add)/2(mul)"
        slice_n1 = ceil_div(slice_n, N0)

        x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m]).to(torch.float32)
        x2 = psb_m1n1m0n0

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

        if(reduce_m_en or reduce_n_en):
            assert(arb_wr_en)
            reduce = Reduce(m1k1m0k0_to_mk(x, slice_m, slice_n), reduce_m_en, reduce_n_en, reduce_mode)

        if (ub_wr_en == True):
            x = m1k1m0k0_to_k1mk0(x, slice_m)
            for n1 in range(slice_n1):
                for m in range(slice_m):
                    for n0 in range(N0):
                        n_idx = n1 * N0 + n0
                        m_idx = m + start_tile_m
                        if(m_idx < tile_m):
                            val = x[n1, m, n0] if n_idx < slice_n else 0
                            if ub_atomic_mode == 0:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] = val
                            elif ub_atomic_mode == 1:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] += val
                            else:
                                ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] *= val

        if(ub_wr_en == True and arb_wr_en == True):
            return ub_n1mn0, reduce
        elif(ub_wr_en == True and arb_wr_en == False):
            return ub_n1mn0
        elif(ub_wr_en == False and arb_wr_en == True):
            return reduce
        else:
            assert False, "invalid configuration"

    # 用于把计算结果搬运到GM, 计算功能都是附带的
    def aru_ub2gm(self, ub_n1mn0, gm_n1mn0, arb, scalar, clamp_min, clamp_max,
                  tile_m, tile_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):

        assert(gm_n1mn0.shape[0] == tensor_n1)
        assert(gm_n1mn0.shape[1] == tensor_m)
        tile_n1 = ceil_div(tile_n, N0)

        if(arb_en == False and scalar_en == False):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0)
            x2 = None
        elif(arb_en == True and scalar_en == False):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0)
            x2 = Broadcast(arb, tile_m, tile_n, br_m, br_n)
        elif(arb_en == False and scalar_en == True):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0)
            x2 = torch.full_like(x1, scalar)
        else:
            assert False, "invalid configuration"

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)
        # Convert back to k1mk0 format for assignment to gm_n1mn0
        x = m1k1m0k0_to_k1mk0(x, tile_m)
        for n1 in range(tile_n1):
            for m in range(tile_m):
                for n0 in range(N0):
                    m_idx = m + start_tensor_m
                    if(m_idx < tensor_m):
                        gm_n1mn0[n1 + start_tensor_n1, m + start_tensor_m, n0] = x[n1, m, n0]
        return gm_n1mn0

    # sqrt(var+eps), 调用aru_arb2arb可以一条指令算完
    def aru_arb2arb(self, arb, scalar, clamp_min, clamp_max,
                  length, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):

        assert(arb.shape[0] == length)

        if(scalar_en):
            x1 = arb
            x2 = torch.full_like(x1, scalar)
        else:
            x1 = arb
            x2 = None

        x1 = x1.to(torch.float32)
        if x2 is not None:
            x2 = x2.to(torch.float32)

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

        return x.to(torch.bfloat16)

    def aru_arb2ub(self, ub_n1mn0, arb, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):

        assert(ub_n1mn0.shape[0] == tile_n1)
        assert(ub_n1mn0.shape[1] == tile_m)
        assert(arb.shape[0] == slice_m or arb.shape[0] == slice_n)

        slice_n1 = ceil_div(slice_n, N0)

        if(scalar_en):
            x1 = arb
            x2 = torch.full_like(x1, scalar)
        else:
            x1 = arb
            x2 = None

        x1 = x1.to(torch.float32)
        if x2 is not None:
            x2 = x2.to(torch.float32)

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)
        x = x.to(torch.bfloat16)

        # arb中数据格式为m1m0
        if(arb.shape[0] == slice_m):
            x = mk_to_k1mk0(x.unsqueeze(-1)) #从m1m0到n1mn0
        else:
            x = mk_to_k1mk0(x.unsqueeze(0)) #从m1m0到n1mn0
        for n1 in range(slice_n1):
            for m in range(slice_m):
                for n0 in range(N0):
                    n_idx = n1 * N0 + n0
                    m_idx = m + start_tile_m
                    if(m_idx < tile_m):
                        val = x[n1, m, n0] if n_idx < slice_n else 0
                        ub_n1mn0[n1 + start_tile_n1, m + start_tile_m, n0] = val

        return ub_n1mn0

    # src: ub, arb.
    # dst: arb
    # 分为broadcast与非broadcast两种
    def aru_ub2arb(self, ub_n1mn0, arb, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                  reduce_m_en, reduce_n_en, reduce_mode):

        assert(ub_n1mn0.shape[0] == tile_n1)
        assert(ub_n1mn0.shape[1] == tile_m)
        if arb_en:
            assert(arb.shape[0] == slice_m or arb.shape[0] == slice_n)

        slice_n1 = ceil_div(slice_n, N0)
        reduce = 0

        # 如果arb_en为true但是br_n与br_m都为false，代表做的是vector的binary，对ub以及arb都不做broadcast
        # 这种情况下按照k1mk0直接计算，会将arb在broadcast中转为k1mk0
        if(arb_en == False and scalar_en == False):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m])
            x2 = None
        elif(arb_en == True and scalar_en == False):
            if(not br_m and not br_n):
                x1 = ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m]
            else:
                x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m])
            x2 = Broadcast(arb, slice_m, slice_n, br_m, br_n)
        elif(arb_en == False and scalar_en == True):
            x1 = k1mk0_to_m1k1m0k0(ub_n1mn0[start_tile_n1:start_tile_n1+slice_n1, start_tile_m:start_tile_m+slice_m])
            x2 = torch.full_like(x1, scalar)
        else:
            assert False, "invalid configuration"

        x1 = x1.to(torch.float32)
        if x2 is not None:
            x2 = x2.to(torch.float32)

        assert(int(add_en) + int(sub_en) + int(max_en) + int(min_en) + int(mul_en) + int(div_en) <= 1), "only one binary op enabled"
        if(x2 is not None):
            x = Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en)
        else:
            x = x1

        x = Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en)

        if(reduce_m_en or reduce_n_en):
            if(not br_m and not br_n and arb_en):
                reduce = Reduce(k1mk0_to_mk(x, slice_n), reduce_m_en, reduce_n_en, reduce_mode)
            else:
                reduce = Reduce(m1k1m0k0_to_mk(x, slice_m, slice_n), reduce_m_en, reduce_n_en, reduce_mode)
            return reduce
        else:
            # 将 n1mn0 或 n1mn0 layout 转换为一维张量
            x = x.to(torch.bfloat16)
            if(not br_m and not br_n and arb_en):
                # x 是 n1mn0 layout，转为二维然后展平
                return k1mk0_to_mk(x, slice_n).reshape(-1)
            else:
                # x 是 m1k1m0k0 layout，转为二维然后展平
                return m1k1m0k0_to_mk(x, slice_m, slice_n).reshape(-1)


    def aru_copy_ub2ub(self, ub_src, length, dsize=2):
        ub_dst = torch.zeros(ub_src.size(), dtype=ub_src.dtype) #n1mn0 layout
        size = 0
        for n1 in range(ub_src.shape[0]):
            for m in range(ub_src.shape[1]):
                for n0 in range(ub_src.shape[2]):
                    val = ub_src[n1][m][n0] if size < length else 0
                    ub_dst[n1][m][n0] = val
                    size += 1
        return ub_dst

    """
    0: GDMA; 1: LDMA; 2: MXU; 3: ARU
    """
    def set_flag(self, src, dst, id):
        if id < MAX_SETWAIT_ID:
            self.set_wait_table[src][dst][id] += 1
        else:
            assert False, "set_wait id out of range"

    def wait_flag(self, src, dst, id):
        if id < MAX_SETWAIT_ID:
            self.set_wait_table[src][dst][id] -= 1
        else:
            assert False, "set_wait id out of range"


    # fence 在pymodel里没有实际功能
    def fence(self, eu):
        pass
