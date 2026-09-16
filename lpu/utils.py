from common import *

def mk_to_k1mk0(mk):
    M = mk.shape[0]
    K = mk.shape[1]
    K1 = ceil_div(K, K0)
    mk1k0 = torch.zeros((M, K1*K0), dtype=mk.dtype)
    mk1k0[:, :K] = mk
    k1mk0 = mk1k0.reshape(M, K1, K0).permute(1, 0, 2)
    return k1mk0

def k1mk0_to_m1k1m0k0(k1mk0):
    K1 = k1mk0.shape[0]
    M = k1mk0.shape[1]
    M1 = ceil_div(M, M0)
    k1m1m0k0 = torch.zeros((K1, M1*M0, K0), dtype=k1mk0.dtype)
    k1m1m0k0[:, :M, :] = k1mk0
    m1k1m0k0 = k1m1m0k0.reshape(K1, M1, M0, K0).permute(1, 0, 2, 3)
    return m1k1m0k0

def k1mk0_to_mk(k1mk0, k):
    mk = torch.zeros((k1mk0.shape[1], k), dtype=k1mk0.dtype)
    mk1k0 = k1mk0.permute(1, 0, 2).reshape(k1mk0.shape[1], k1mk0.shape[0]*k1mk0.shape[2])
    mk= mk1k0[:, :k]
    return mk

def m1k1m0k0_to_k1mk0(m1k1m0k0, m):
    M1 = m1k1m0k0.shape[0]
    K1 = m1k1m0k0.shape[1]
    M0 = m1k1m0k0.shape[2]
    K0 = m1k1m0k0.shape[3]
    k1m1m0k0 = m1k1m0k0.permute(1, 0, 2, 3).reshape(K1, M1*M0, K0)
    k1mk0 = k1m1m0k0[:, :m] 
    return k1mk0

def m1k1m0k0_to_mk(m1k1m0k0, m, k):
    M1 = m1k1m0k0.shape[0]
    K1 = m1k1m0k0.shape[1]
    M0 = m1k1m0k0.shape[2]
    K0 = m1k1m0k0.shape[3]
    m1m0k1k0 = m1k1m0k0.permute(0, 2, 1, 3).reshape(M1*M0, K1*K0)
    mk = m1m0k1k0[:m, :k]
    return mk

def generate_matmul_params():
    M_L2 = np.random.randint(3, 100)
    N_L2 = np.random.randint(3, 100)
    K_L2 = ceil_align(np.random.randint(3, 100), K0)
    matmul_params = {
        'M': M_L2,
        'N': N_L2,
        'K': K_L2,
    }
    print(matmul_params)
    return matmul_params

def generate_matmul_tensor(matmul_params, dtype=torch.bfloat16):
    M = matmul_params['M']
    N = matmul_params['N']
    K = matmul_params['K']
    left = torch.randn((M, K), dtype=dtype)
    right = torch.randn((N, K), dtype=dtype)
    bias = torch.randn((N), dtype=torch.float32)
    return left, right, bias

def compare(tensor_test, tensor_golden, int = False, golden_threshold = 0.001, error_threshold = 0.1):
    # 统一转成 torch.Tensor
    if not isinstance(tensor_test, torch.Tensor):
        tensor_test = torch.as_tensor(tensor_test, dtype=torch.bfloat16)
    if not isinstance(tensor_golden, torch.Tensor):
        tensor_golden = torch.as_tensor(tensor_golden, dtype=torch.bfloat16)

    if int:
        diff = torch.abs(tensor_test.flatten() - tensor_golden.flatten())
        error_rate = diff / (torch.abs(tensor_golden.flatten()) + 1)
        print("average error rate: {:.4f}%".format(error_rate.mean()*100))
        if error_rate.mean() < error_threshold:
            print("pass")
            return True
        else:
            print("fail")
            return False
    else:
        # 受限于float32的计算精度，golden的值太小，相对误差可能很大
        golden_mask = torch.abs(tensor_golden) > golden_threshold
        diff = torch.abs(tensor_test[golden_mask].flatten() - tensor_golden[golden_mask].flatten())
        error_rate = diff / (torch.abs(tensor_golden[golden_mask].flatten()) + 1e-10)
        print("average error rate: {:.4f}%".format(error_rate.mean()*100))
        if error_rate.mean() < error_threshold:
            print("pass")
            return True
        else:
            print(diff, tensor_golden[golden_mask])
            print("fail")
            return False


def compare_results(op_name, result, golden):
    if op_name == "qwen3_prefill_attention":
        out, k_cache, v_cache = result
        golden_out, golden_k_cache, golden_v_cache = golden
        return (
            torch.allclose(out.float(), golden_out.float(), atol=4e-2, rtol=0.1)
            and torch.allclose(k_cache.float(), golden_k_cache.float(), atol=4e-2, rtol=0.05)
            and torch.allclose(v_cache.float(), golden_v_cache.float(), atol=4e-2, rtol=0.05)
        )
    return compare(result, golden)

def ceil_div(x, y):
    return (x + y - 1) // y

def ceil_align(x, y):
    return ceil_div(x, y) * y

def sizeof(tensor, dtype):
    return tensor.nelement() * dtype

def int8_to_uint8(value):
    if value < 0:
        return value + 256
    else:
        return value

def float32_to_bytes(value):
    num_bytes = 4
    float_bytes = struct.pack('<f', float(value))  # 将float打包成字节，使用小端序
    np_value = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(num_bytes):  # float是4字节，所以最多取4个字节
        np_value[i] = float_bytes[i]
    return np_value

def float32_to_float16_bytes(value):
    num_bytes = 2
    float_bytes = struct.pack('<f', float(value))  # 将float打包成字节，使用小端序
    np_value = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(num_bytes):  # float16是2字节，所以最多取2个字节
        np_value[i] = float_bytes[i]
    return np_value

def float16_to_bytes(value):
    num_bytes = 2
    float_bytes = struct.pack('<e', np.float16(value))  # 将float打包成字节，使用小端序
    np_value = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(num_bytes):  # float16是2字节，所以最多取2个字节
        np_value[i] = float_bytes[i]
    return np_value


# def bfloat16_to_bytes(value):
#     num_bytes = 2
#     bf16_bits = int(torch.tensor([float(value)], dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).item())
#     bf16_bytes = bf16_bits.to_bytes(num_bytes, byteorder='little')
#     np_value = np.zeros(num_bytes, dtype=np.uint8)
#     for i in range(num_bytes):
#         np_value[i] = bf16_bytes[i]
#     return np_value


def int_to_bytes(num_bytes, value):
    num_bits = np.int64(num_bytes*8)
    max_value = 2**num_bits - 1
    clip_value = int(np.clip(value, 0, max_value))
    np_value = np.zeros(num_bytes, dtype = np.uint8)
    value_tobytes = clip_value.to_bytes(num_bytes, byteorder = 'little')
    for i in range(num_bytes):
        np_value[i] = value_tobytes[i]
    return np_value

def float32_to_int(value):
    value_tobytes = struct.pack('f', value)
    integer = int.from_bytes(value_tobytes, byteorder = 'little')
    return integer

def bfloat16_to_bytes(value):
    num_bytes = 2
    bf16_bits = torch.tensor(value, dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF
    np_value = np.zeros(num_bytes, dtype=np.uint8)
    value_tobytes = int(bf16_bits).to_bytes(num_bytes, byteorder='little')
    for i in range(num_bytes):
        np_value[i] = value_tobytes[i]
    return np_value
