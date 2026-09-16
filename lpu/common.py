import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import struct

M0 = 16
C0 = N0 = 8
K0_Byte = 16
K0 = 8
BUS_WIDTH = 128
UB_ALIGN = K0_Byte
LMB_ALIGN = M0 * K0_Byte
RMB_ALIGN = N0 * K0_Byte
PMB_ALIGN = K0_Byte
PSB_ALIGN = M0 * N0 * 4
ARB_ALIGN = 16

GM_SIZE  = 512 * 1024 * 1024
UB_SIZE  = 1024 * 1024
LMB_SIZE = 64  * 1024
RMB_SIZE = 64  * 1024
PMB_SIZE = 16  * 1024
PSB_SIZE = 256 * 1024
ARB_SIZE = 2   * 1024
MAX_SETWAIT_ID = 16
HLAF_ID = MAX_SETWAIT_ID // 2
