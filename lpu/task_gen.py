from common import *
from utils import *

class TaskGen:
    def __init__(self, task_name):
        self.task_name = task_name
        self.instr_bin = bytearray()
        self.instr_ptx = []  # 保存指令的文本描述
        self.instr_size = 32  # 固定指令大小为32B
        self.instr_idx = 0  # 自动维护指令索引
        self.input_data_bin = bytearray()
        self.output_data_bin = bytearray()

        # Head信息
        self.task_head = {
            'instr_size': 0,   # 指令段大小
            'input_size': 0,   # 输入数据段大小
            'output_size': 0   # 输出数据段大小
        }

    def add_input_data(self, data):
        if data.dtype == torch.bfloat16:
            # bf16需要特殊处理，通过view转换为uint16保持二进制表示
            self.input_data_bin.extend(data.view(torch.uint16).numpy().tobytes())
        else:
            # 其他数据类型正常处理
            self.input_data_bin.extend(data.numpy().tobytes())

    def add_output_data(self, data):
        if data.dtype == torch.bfloat16:
            # bf16需要特殊处理，通过view转换为uint16保持二进制表示
            self.output_data_bin.extend(data.view(torch.uint16).numpy().tobytes())
        else:
            # 其他数据类型正常处理
            self.output_data_bin.extend(data.numpy().tobytes())

    def generate(self):
        ptx = open("{}.ptx".format(self.task_name), "w")
        for instr in self.instr_ptx:
            ptx.write(instr + "\n")
        ptx.close()

        instr_bin = np.array(self.instr_bin)
        input_data_bin  = np.array(self.input_data_bin)
        output_data_bin = np.array(self.output_data_bin)
        head = np.zeros(3, dtype=np.int32)
        head[0] = instr_bin.size
        head[1] = input_data_bin.size
        head[2] = output_data_bin.size

        # 组合所有数据并写入二进制文件
        bin_file = open("{}.bin".format(self.task_name), "wb")
        bin_file.write(head.tobytes())
        bin_file.write(instr_bin.tobytes())
        bin_file.write(input_data_bin.tobytes())
        bin_file.write(output_data_bin.tobytes())
        bin_file.close()

    def gdma_gm2ub(self, gm_addr, ub_addr, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, tile_m, tile_k1):
        ptx = "gdma_gm2ub: opcode 0, instr_idx {}, gm_addr {}, ub_addr {}, tensor_m {}, start_tensor_m {}, tensor_k1 {}, start_tensor_k1 {}, tile_m {}, tile_k1 {}".\
        format(self.instr_idx, gm_addr, ub_addr, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, tile_m, tile_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 0
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:8] = int_to_bytes(5, gm_addr)
        instr[8:11] = int_to_bytes(3, ub_addr)
        instr[11:13] = int_to_bytes(2, tensor_m)
        instr[13:15] = int_to_bytes(2, start_tensor_m)
        instr[15:17] = int_to_bytes(2, tensor_k1)
        instr[17:19] = int_to_bytes(2, start_tensor_k1)
        instr[19:21] = int_to_bytes(2, tile_m)
        instr[21:23] = int_to_bytes(2, tile_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def gdma_gm2lmb(self, gm_addr, lmb_addr, tensor_m, start_tensor_m, tensor_k1, start_tensor_k1, slice_m, slice_k1):
        ptx = "gdma_gm2lmb: opcode 1, instr_idx {}, gm_addr {}, lmb_addr {}, tensor_m {}, start_tensor_m {}, start_tensor_k1 {}, slice_m {}, slice_k1 {}".\
        format(self.instr_idx, gm_addr, lmb_addr, tensor_m, start_tensor_m, start_tensor_k1, slice_m, slice_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 1
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:8] = int_to_bytes(5, gm_addr)
        instr[8:10] = int_to_bytes(2, lmb_addr)
        instr[10:12] = int_to_bytes(2, tensor_m)
        instr[12:14] = int_to_bytes(2, start_tensor_m)
        instr[14:16] = int_to_bytes(2, tensor_k1)
        instr[16:18] = int_to_bytes(2, start_tensor_k1)
        instr[18:20] = int_to_bytes(2, slice_m)
        instr[20:22] = int_to_bytes(2, slice_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def gdma_gm2rmb(self, gm_addr, rmb_addr, tensor_n, start_tensor_n, tensor_k1, start_tensor_k1, slice_n, slice_k1):
        ptx = "gdma_gm2rmb: opcode 2, instr_idx {}, gm_addr {}, rmb_addr {}, tensor_n {}, start_tensor_n {}, start_tensor_k1 {}, slice_n {}, slice_k1 {}".\
        format(self.instr_idx, gm_addr, rmb_addr, tensor_n, start_tensor_n, start_tensor_k1, slice_n, slice_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 2
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:8] = int_to_bytes(5, gm_addr)
        instr[8:10] = int_to_bytes(2, rmb_addr)
        instr[10:12] = int_to_bytes(2, tensor_n)
        instr[12:14] = int_to_bytes(2, start_tensor_n)
        instr[14:16] = int_to_bytes(2, tensor_k1)
        instr[16:18] = int_to_bytes(2, start_tensor_k1)
        instr[18:20] = int_to_bytes(2, slice_n)
        instr[20:22] = int_to_bytes(2, slice_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def gdma_gm2pmb(self, gm_addr, pmb_addr, tensor_n, start_tensor_n, slice_n):
        ptx = "gdma_gm2pmb: opcode 3, instr_idx {}, gm_addr {}, pmb_addr {}, tensor_n {}, start_tensor_n {}, slice_n {}".\
        format(self.instr_idx, gm_addr, pmb_addr, tensor_n, start_tensor_n, slice_n)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 3
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:8] = int_to_bytes(5, gm_addr)
        instr[8:10] = int_to_bytes(2, pmb_addr)
        instr[10:12] = int_to_bytes(2, tensor_n)
        instr[12:14] = int_to_bytes(2, start_tensor_n)
        instr[14:16] = int_to_bytes(2, slice_n)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def gdma_zero2ub(self, ub_addr, tile_m, tile_k1):
        ptx = "gdma_zero2ub: opcode 4, instr_idx {}, ub_addr {}, tile_m {}, tile_k1 {}".\
        format(self.instr_idx, ub_addr, tile_m, tile_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 4
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6:8] = int_to_bytes(2, tile_m)
        instr[8:10] = int_to_bytes(2, tile_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def gdma_scalar2ub(self, ub_addr, tile_m, tile_k1, scalar):
        ptx = "gdma_scalar2ub: opcode 5, instr_idx {}, ub_addr {}, tile_m {}, tile_k1 {}, scalar {}".\
        format(self.instr_idx, ub_addr, tile_m, tile_k1, scalar)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 5
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6:8] = int_to_bytes(2, tile_m)
        instr[8:10] = int_to_bytes(2, tile_k1)
        instr[10:12] = bfloat16_to_bytes(scalar) #默认返回2bytes
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def ldma_ub2lmb(self, ub_addr, lmb_addr, tile_m, start_tile_m, tile_k1, start_tile_k1, slice_m, slice_k1):
        ptx = "ldma_ub2lmb: opcode 16, instr_idx {}, ub_addr {}, lmb_addr {}, tile_m {}, start_tile_m {}, tile_k1 {}, start_tile_k1 {}, slice_m {}, slice_k1 {}".\
        format(self.instr_idx, ub_addr, lmb_addr, tile_m, start_tile_m, tile_k1, start_tile_k1, slice_m, slice_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 16
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6:8] = int_to_bytes(2, lmb_addr)
        instr[8:10] = int_to_bytes(2, tile_m)
        instr[10:12] = int_to_bytes(2, start_tile_m)
        instr[12:14] = int_to_bytes(2, tile_k1)
        instr[14:16] = int_to_bytes(2, start_tile_k1)
        instr[16:18] = int_to_bytes(2, slice_m)
        instr[18:20] = int_to_bytes(2, slice_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def ldma_ub2rmb(self, ub_addr, rmb_addr, tile_n, start_tile_n, tile_k1, start_tile_k1, slice_n, slice_k1):
        ptx = "ldma_ub2rmb: opcode 17, instr_idx {}, ub_addr {}, rmb_addr {}, tile_n {}, start_tile_n {}, tile_k1 {}, start_tile_k1 {}, slice_n {}, slice_k1 {}".\
        format(self.instr_idx, ub_addr, rmb_addr, tile_n, start_tile_n, tile_k1, start_tile_k1, slice_n, slice_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 17
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6:8] = int_to_bytes(2, rmb_addr)
        instr[8:10] = int_to_bytes(2, tile_n)
        instr[10:12] = int_to_bytes(2, start_tile_n)
        instr[12:14] = int_to_bytes(2, tile_k1)
        instr[14:16] = int_to_bytes(2, start_tile_k1)
        instr[16:18] = int_to_bytes(2, slice_n)
        instr[18:20] = int_to_bytes(2, slice_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def ldma_ub2rmb_transpose(self, ub_addr, rmb_addr, tile_n, start_tile_n, tile_k1, start_tile_k1, slice_n, slice_k1):
        ptx = "ldma_ub2rmb_transpose: opcode 18, instr_idx {}, ub_addr {}, rmb_addr {}, tile_n {}, start_tile_n {}, tile_k1 {}, start_tile_k1 {}, slice_n {}, slice_k1 {}".\
        format(self.instr_idx, ub_addr, rmb_addr, tile_n, start_tile_n, tile_k1, start_tile_k1, slice_n, slice_k1)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 18
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6:8] = int_to_bytes(2, rmb_addr)
        instr[8:10] = int_to_bytes(2, tile_n)
        instr[10:12] = int_to_bytes(2, start_tile_n)
        instr[12:14] = int_to_bytes(2, tile_k1)
        instr[14:16] = int_to_bytes(2, start_tile_k1)
        instr[16:18] = int_to_bytes(2, slice_n)
        instr[18:20] = int_to_bytes(2, slice_k1)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def mxu_matmul(self, lmb_addr, rmb_addr, pmb_addr, psb_addr, slice_m, slice_n, slice_k1, bias_en, psum_en):
        ptx = "mxu_matmul: opcode 32, instr_idx {}, lmb_addr {}, rmb_addr {}, pmb_addr {}, psb_addr {}, slice_m {}, slice_n {}, slice_k1 {}, bias_en {}, psum_en {}".\
        format(self.instr_idx, lmb_addr, rmb_addr, pmb_addr, psb_addr, slice_m, slice_n, slice_k1, int(bias_en), int(psum_en))
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 32
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:5] = int_to_bytes(2, lmb_addr)
        instr[5:7] = int_to_bytes(2, rmb_addr)
        instr[7:9] = int_to_bytes(2, pmb_addr)
        instr[9:12] = int_to_bytes(3, psb_addr)
        instr[12:14] = int_to_bytes(2, slice_m)
        instr[14:16] = int_to_bytes(2, slice_n)
        instr[16:18] = int_to_bytes(2, slice_k1)
        instr[18] = (int(psum_en) << 1) | int(bias_en)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_psb2ub(self, psb_addr, ub_addr, arb_addr, scalar, clamp_min, clamp_max,
                   slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                   arb_en, br_m, br_n, scalar_en,
                   add_en, sub_en, max_en, min_en, mul_en, div_en,
                   neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                   reduce_m_en, reduce_n_en, reduce_mode,
                   ub_wr_en, arb_wr_en, ub_atomic_mode=0):
        ptx = "aru_psb2ub: opcode 48, instr_idx {}, psb_addr {}, ub_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tile_m {}, tile_n1 {}, start_tile_m {}, start_tile_n1 {}, arb_en {}, br_m {}, br_n {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}, reduce_m_en {}, reduce_n_en {}, reduce_mode {}, ub_wr_en {}, arb_wr_en {}, ub_atomic_mode {}".\
        format(self.instr_idx, psb_addr, ub_addr, arb_addr, scalar, clamp_min, clamp_max, slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, int(arb_en), int(br_m), int(br_n), int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en), int(reduce_m_en), int(reduce_n_en), int(reduce_mode), int(ub_wr_en), int(arb_wr_en), int(ub_atomic_mode))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 48
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, psb_addr)
        instr[6:9] = int_to_bytes(3, ub_addr)
        instr[9]   = arb_addr
        instr[10:12] = bfloat16_to_bytes(scalar)
        instr[12:14] = bfloat16_to_bytes(clamp_min)
        instr[14:16] = bfloat16_to_bytes(clamp_max)
        instr[16:18] = int_to_bytes(2, slice_m)
        instr[18:20] = int_to_bytes(2, slice_n)
        instr[20:22] = int_to_bytes(2, tile_m)
        instr[22:24] = int_to_bytes(2, tile_n1)
        instr[24:26] = int_to_bytes(2, start_tile_m)
        instr[26:28] = int_to_bytes(2, start_tile_n1)
        instr[28] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(scalar_en) << 3) |  (int(br_n) << 2) | (int(br_m) << 1) | int(arb_en)
        instr[29] = (int(recp_en) << 7) | (int(pow_en) << 6) | (int(sqrt_en) << 5) | (int(exp_en) << 4) |  (int(clamp_en) << 3) | (int(neg_en) << 2) | int(div_en << 1) | int(mul_en)
        instr[30] = (int(ub_atomic_mode) << 6) | (int(arb_wr_en) << 5) | (int(ub_wr_en) << 4) | (int(reduce_mode) << 2) | int(reduce_n_en << 1) | (int(reduce_m_en))

        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_psb2gm(self, psb_addr, gm_addr, arb_addr, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):
        ptx = "aru_psb2gm: opcode 49, instr_idx {}, psb_addr {}, gm_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tensor_m {}, tensor_n1 {}, start_tensor_m {}, start_tensor_n1 {}, arb_en {}, br_m {}, br_n {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}".\
        format(self.instr_idx, psb_addr, gm_addr, arb_addr, scalar, clamp_min, clamp_max, slice_m, slice_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1, int(arb_en), int(br_m), int(br_n), int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 49
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        # 修改后的布局: opcode | instr_idx | gm_addr(5B) | psb_addr(3B) | ...
        instr[3:8]  = int_to_bytes(5, gm_addr)
        instr[8:11] = int_to_bytes(3, psb_addr)
        instr[11]   = arb_addr
        instr[12:14] = bfloat16_to_bytes(scalar)
        instr[14:16] = bfloat16_to_bytes(clamp_min)
        instr[16:18] = bfloat16_to_bytes(clamp_max)
        instr[18:20] = int_to_bytes(2, slice_m)
        instr[20:22] = int_to_bytes(2, slice_n)
        instr[22:24] = int_to_bytes(2, tensor_m)
        instr[24:26] = int_to_bytes(2, tensor_n1)
        instr[26:28] = int_to_bytes(2, start_tensor_m)
        instr[28:30] = int_to_bytes(2, start_tensor_n1)
        instr[30] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(scalar_en) << 3) |  (int(br_n) << 2) | (int(br_m) << 1) | int(arb_en)
        instr[31] = (int(recp_en) << 6) | (int(pow_en) << 5) | (int(sqrt_en) << 4) | (int(exp_en) << 3) |  (int(clamp_en) << 2) | (int(neg_en) << 1) | int(div_en)

        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_ub2ub(self, ub_addr, arb_addr, scalar, clamp_min, clamp_max,
                  slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                  reduce_m_en, reduce_n_en, reduce_mode,
                  ub_wr_en, arb_wr_en, ub_atomic_mode=0):
        ptx = "aru_ub2ub: opcode 50, instr_idx {}, ub_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tile_m {}, tile_n1 {}, start_tile_m {}, start_tile_n1 {}, arb_en {}, br_m {}, br_n {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}, reduce_m_en {}, reduce_n_en {}, reduce_mode {}, ub_wr_en {}, arb_wr_en {}, ub_atomic_mode {}".\
        format(self.instr_idx, ub_addr, arb_addr, scalar, clamp_min, clamp_max, slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, int(arb_en), int(br_m), int(br_n), int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en), int(reduce_m_en), int(reduce_n_en), int(reduce_mode), int(ub_wr_en), int(arb_wr_en), int(ub_atomic_mode))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 50
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6]   = arb_addr
        instr[7:9] = bfloat16_to_bytes(scalar)
        instr[9:11] = bfloat16_to_bytes(clamp_min)
        instr[11:13] = bfloat16_to_bytes(clamp_max)
        instr[13:15] = int_to_bytes(2, slice_m)
        instr[15:17] = int_to_bytes(2, slice_n)
        instr[17:19] = int_to_bytes(2, tile_m)
        instr[19:21] = int_to_bytes(2, tile_n1)
        instr[21:23] = int_to_bytes(2, start_tile_m)
        instr[23:25] = int_to_bytes(2, start_tile_n1)
        instr[25] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(scalar_en) << 3) |  (int(br_n) << 2) | (int(br_m) << 1) | int(arb_en)
        instr[26] = (int(recp_en) << 7) | (int(pow_en) << 6) | (int(sqrt_en) << 5) | (int(exp_en) << 4) |  (int(clamp_en) << 3) | (int(neg_en) << 2) | int(div_en << 1) | int(mul_en)
        instr[27] = (int(ub_atomic_mode) << 6) | (int(arb_wr_en) << 5) | (int(ub_wr_en) << 4) | (int(reduce_mode) << 2) | int(reduce_n_en << 1) |(int(reduce_m_en))

        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_dual2ub(self, arb_addr, psb_addr, ub_addr, clamp_min, clamp_max,
                    slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                    add_en, sub_en, max_en, min_en, mul_en, div_en,
                    neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                    reduce_m_en, reduce_n_en, reduce_mode,
                    ub_wr_en, arb_wr_en, ub_atomic_mode=0):
        ptx = "aru_dual2ub: opcode 51, instr_idx {}, arb_addr {}, psb_addr {}, ub_addr {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tile_m {}, tile_n1 {}, start_tile_m {}, start_tile_n1 {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}, reduce_m_en {}, reduce_n_en {}, reduce_mode {}, ub_wr_en {}, arb_wr_en {}, ub_atomic_mode {}".\
        format(self.instr_idx, arb_addr, psb_addr, ub_addr, clamp_min, clamp_max, slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en), int(reduce_m_en), int(reduce_n_en), int(reduce_mode), int(ub_wr_en), int(arb_wr_en), int(ub_atomic_mode))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 51
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3]   = arb_addr
        instr[4:7] = int_to_bytes(3, psb_addr)
        instr[7:10] = int_to_bytes(3, ub_addr)
        instr[10:12] = bfloat16_to_bytes(clamp_min)
        instr[12:14] = bfloat16_to_bytes(clamp_max)
        instr[14:16] = int_to_bytes(2, slice_m)
        instr[16:18] = int_to_bytes(2, slice_n)
        instr[18:20] = int_to_bytes(2, tile_m)
        instr[20:22] = int_to_bytes(2, tile_n1)
        instr[22:24] = int_to_bytes(2, start_tile_m)
        instr[24:26] = int_to_bytes(2, start_tile_n1)
        instr[26] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(div_en) << 3) | (int(mul_en) << 2)
        instr[27] = (int(reduce_m_en) << 7) | (int(recp_en) << 6) | (int(pow_en) << 5) | (int(sqrt_en) << 4) | (int(exp_en) << 3) |  (int(clamp_en) << 2) | (int(neg_en) << 1)
        instr[28] = (int(ub_atomic_mode) << 5) | (int(arb_wr_en) << 4) | (int(ub_wr_en) << 3) | (int(reduce_mode) << 1) | int(reduce_n_en)

        self.instr_bin.extend(instr)
        self.instr_idx += 1

    # 与C++中定义不同，修正
    def aru_ub2gm(self, ub_addr, gm_addr, arb_addr, scalar, clamp_min, clamp_max,
                  tile_m, tile_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1,
                  arb_en, br_m, br_n, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):
        ptx = "aru_ub2gm: opcode 52, instr_idx {}, ub_addr {}, gm_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, tile_m {}, tile_n {}, tensor_m {}, tensor_n1 {}, start_tensor_m {}, start_tensor_n1 {}, arb_en {}, br_m {}, br_n {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}".\
        format(self.instr_idx, ub_addr, gm_addr, arb_addr, scalar, clamp_min, clamp_max, tile_m, tile_n, tensor_m, tensor_n1, start_tensor_m, start_tensor_n1, int(arb_en), int(br_m), int(br_n), int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en))
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 52
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        # 统一为 gm_addr 5B：布局与 aru_psb2gm 对齐
        # opcode | instr_idx | gm_addr(5B) | ub_addr(3B) | arb_addr(1B) | ...
        instr[3:8]  = int_to_bytes(5, gm_addr)
        instr[8:11] = int_to_bytes(3, ub_addr)
        instr[11]   = arb_addr
        instr[12:14] = bfloat16_to_bytes(scalar)
        instr[14:16] = bfloat16_to_bytes(clamp_min)
        instr[16:18] = bfloat16_to_bytes(clamp_max)
        instr[18:20] = int_to_bytes(2, tile_m)
        instr[20:22] = int_to_bytes(2, tile_n)
        instr[22:24] = int_to_bytes(2, tensor_m)
        instr[24:26] = int_to_bytes(2, tensor_n1)
        instr[26:28] = int_to_bytes(2, start_tensor_m)
        instr[28:30] = int_to_bytes(2, start_tensor_n1)
        instr[30] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(scalar_en) << 3) |  (int(br_n) << 2) | (int(br_m) << 1) | int(arb_en)
        instr[31] = (int(recp_en) << 7) | (int(pow_en) << 6) | (int(sqrt_en) << 5) | (int(exp_en) << 4) |  (int(clamp_en) << 3) | (int(neg_en) << 2) | int(div_en << 1) | int(mul_en)

        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_arb2arb(self, arb_addr, scalar, clamp_min, clamp_max,
                  length, scalar_en,
                  add_en, sub_en, max_en, min_en, mul_en, div_en,
                  neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):
        ptx = "aru_arb2arb: opcode 53, instr_idx {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, length {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}".\
        format(self.instr_idx, arb_addr, scalar, clamp_min, clamp_max, length, int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en))
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 53
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, arb_addr)
        instr[6:8] = bfloat16_to_bytes(scalar)
        instr[8:10] = bfloat16_to_bytes(clamp_min)
        instr[10:12] = bfloat16_to_bytes(clamp_max)
        instr[12:14] = int_to_bytes(2, length)
        instr[14] = (int(neg_en) << 7) | (int(div_en) << 6) | (int(mul_en) << 5) | (int(min_en) << 4) | (int(max_en) << 3) | (int(sub_en) << 2) | (int(add_en) << 1) | int(scalar_en)
        instr[15] = (int(recp_en) << 4) | (int(pow_en) << 3) | (int(sqrt_en) << 2) | (int(exp_en) << 1) | int(clamp_en)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_ub2arb(self, ub_addr, arb_addr, scalar, clamp_min, clamp_max,
                   slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                   arb_en, br_m, br_n, scalar_en,
                   add_en, sub_en, max_en, min_en, mul_en, div_en,
                   neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en,
                   reduce_m_en, reduce_n_en, reduce_mode):
        ptx = "aru_ub2arb: opcode 54, instr_idx {}, ub_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tile_m {}, tile_n1 {}, start_tile_m {}, start_tile_n1 {}, arb_en {}, br_m {}, br_n {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}, reduce_m_en {}, reduce_n_en {}, reduce_mode {}".\
        format(self.instr_idx, ub_addr, arb_addr, scalar, clamp_min, clamp_max, slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, int(arb_en), int(br_m), int(br_n), int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en), int(reduce_m_en), int(reduce_n_en), int(reduce_mode))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 54
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6]   = arb_addr
        instr[7:9] = bfloat16_to_bytes(scalar)
        instr[9:11] = bfloat16_to_bytes(clamp_min)
        instr[11:13] = bfloat16_to_bytes(clamp_max)
        instr[13:15] = int_to_bytes(2, slice_m)
        instr[15:17] = int_to_bytes(2, slice_n)
        instr[17:19] = int_to_bytes(2, tile_m)
        instr[19:21] = int_to_bytes(2, tile_n1)
        instr[21:23] = int_to_bytes(2, start_tile_m)
        instr[23:25] = int_to_bytes(2, start_tile_n1)
        instr[25] = (int(min_en) << 7) | (int(max_en) << 6) | (int(sub_en) << 5) | (int(add_en) << 4) | (int(scalar_en) << 3) |  (int(br_n) << 2) | (int(br_m) << 1) | int(arb_en)
        instr[26] = (int(recp_en) << 7) | (int(pow_en) << 6) | (int(sqrt_en) << 5) | (int(exp_en) << 4) |  (int(clamp_en) << 3) | (int(neg_en) << 2) | int(div_en << 1) | int(mul_en)
        instr[27] = (int(reduce_mode) << 2) | int(reduce_n_en << 1) |(int(reduce_m_en))
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_arb2ub(self, ub_addr, arb_addr, scalar, clamp_min, clamp_max,
                   slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1,
                   scalar_en, add_en, sub_en, max_en, min_en, mul_en, div_en,
                   neg_en, clamp_en, exp_en, sqrt_en, pow_en, recp_en):
        ptx = "aru_arb2ub: opcode 55, instr_idx {}, ub_addr {}, arb_addr {}, scalar {}, clamp_min {}, clamp_max {}, slice_m {}, slice_n {}, tile_m {}, tile_n1 {}, start_tile_m {}, start_tile_n1 {}, scalar_en {}, add_en {}, sub_en {}, max_en {}, min_en {}, mul_en {}, div_en {}, neg_en {}, clamp_en {}, exp_en {}, sqrt_en {}, pow_en {}, recp_en {}".\
        format(self.instr_idx, ub_addr, arb_addr, scalar, clamp_min, clamp_max, slice_m, slice_n, tile_m, tile_n1, start_tile_m, start_tile_n1, int(scalar_en), int(add_en), int(sub_en), int(max_en), int(min_en), int(mul_en), int(div_en), int(neg_en), int(clamp_en), int(exp_en), int(sqrt_en), int(pow_en), int(recp_en))
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 55
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, ub_addr)
        instr[6]   = arb_addr
        instr[7:9] = bfloat16_to_bytes(scalar)
        instr[9:11] = bfloat16_to_bytes(clamp_min)
        instr[11:13] = bfloat16_to_bytes(clamp_max)
        instr[13:15] = int_to_bytes(2, slice_m)
        instr[15:17] = int_to_bytes(2, slice_n)
        instr[17:19] = int_to_bytes(2, tile_m)
        instr[19:21] = int_to_bytes(2, tile_n1)
        instr[21:23] = int_to_bytes(2, start_tile_m)
        instr[23:25] = int_to_bytes(2, start_tile_n1)
        instr[25] =  (int(neg_en) << 7) | int(div_en << 6) | int(mul_en << 5) | (int(min_en) << 4) | (int(max_en) << 3) | (int(sub_en) << 2) | (int(add_en) << 1) | int(scalar_en)
        instr[26] =  (int(recp_en) << 4) | (int(pow_en) << 3) | (int(sqrt_en) << 2) | (int(exp_en) << 1) | int(clamp_en)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def aru_copy_ub2ub(self, src_addr, dst_addr, length):
        ptx = "aru_copy_ub2ub: opcode 56, instr_idx {}, src_addr {}, dst_addr {}, length {}".format(self.instr_idx, src_addr, dst_addr, length)
        self.instr_ptx.append(ptx)
        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 56
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3:6] = int_to_bytes(3, src_addr)
        instr[6:9] = int_to_bytes(3, dst_addr)
        instr[9:12] = int_to_bytes(3, length)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def set_flag(self, src, dst, id):
        ptx = "set_flag: opcode 64, instr_idx {}, src {}, dst {}, id {}".format(self.instr_idx, src, dst, id)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 64
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3] = src
        instr[4] = dst
        instr[5] = id
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def wait_flag(self, src, dst, id):
        ptx = "wait_flag: opcode 65, instr_idx {}, src {}, dst {}, id {}".format(self.instr_idx, src, dst, id)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 65
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3] = src
        instr[4] = dst
        instr[5] = id
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def fence(self, eu):
        ptx = "fence: opcode 66, instr_idx {}, eu {}".format(self.instr_idx, eu)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 66
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        instr[3] = eu
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def start(self):
        ptx = "start: opcode 67, instr_idx {}".format(self.instr_idx)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 67
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        self.instr_bin.extend(instr)
        self.instr_idx += 1

    def end(self):
        ptx = "end: opcode 68, instr_idx {}".format(self.instr_idx)
        self.instr_ptx.append(ptx)

        instr = np.zeros(self.instr_size, dtype = np.uint8)
        instr[0] = 68
        instr[1:3] = int_to_bytes(2, self.instr_idx)
        self.instr_bin.extend(instr)
        self.instr_idx += 1
