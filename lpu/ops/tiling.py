from utils import *

def find_optimal_l1_tiling(M_L2, N_L2, K_L2):
    M_L1 = ceil_align(np.random.randint(2, M_L2), M0)
    N_L1 = ceil_align(np.random.randint(2, N_L2), N0)
    K_L1 = ceil_align(np.random.randint(2, K_L2), K0)
    return M_L1, N_L1, K_L1

def find_optimal_l0_tiling(M_L1, N_L1, K_L1):
    M_L0 = ceil_align(np.random.randint(1, M_L1), M0) #不改成1，则可能会出现 2 = M_L1 的情况，low >= high
    N_L0 = ceil_align(np.random.randint(1, N_L1), N0)
    K_L0 = ceil_align(np.random.randint(1, K_L1), K0)
    return M_L0, N_L0, K_L0


def resolve_matmul_tiling(M, N, K, l1_tile=None, l0_tile=None):
    """Resolve random/default or explicit deterministic matmul tiling."""
    M_L1, N_L1, K_L1 = (
        find_optimal_l1_tiling(M, N, K) if l1_tile is None else l1_tile
    )
    M_L0, N_L0, K_L0 = (
        find_optimal_l0_tiling(M_L1, N_L1, K_L1)
        if l0_tile is None
        else l0_tile
    )

    assert all(value > 0 for value in (M_L1, N_L1, K_L1, M_L0, N_L0, K_L0))
    assert M_L1 % M0 == 0 and N_L1 % N0 == 0 and K_L1 % K0 == 0
    assert M_L0 % M0 == 0 and N_L0 % N0 == 0 and K_L0 % K0 == 0
    assert M_L0 <= M_L1 and N_L0 <= N_L1 and K_L0 <= K_L1
    return M_L1, N_L1, K_L1, M_L0, N_L0, K_L0
