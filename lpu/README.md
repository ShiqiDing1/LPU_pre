# LPU Tutorial 使用说明

为了让初学者循序渐进地理解 LPU 的矩阵分块、data layout、ISA 指令语义、TaskGen 地址计算以及多执行单元同步，本教程使用一个 tiled matmul 贯穿全部练习。

请按照 Part 1 → Part 2 → Part 3 的顺序调试代码。每完成一个 Part，先运行对应测试并确认通过，再进入下一阶段。

~~~bash
python3 tests/test_part1.py
python3 tests/test_part2.py
python3 tests/test_part3.py
~~~

每个阶段的达标信号分别为：

~~~text
[Part1] PASS
[Part2] PASS
[Part3] PASS
~~~

测试只比较最终结果，不会指出需要修改的具体代码位置。调试过程中请阅读并理解现有实现，不要使用 `torch.matmul` 替换练习中的 LPU 算子流程；PyTorch 只在测试中用于生成 golden。

## 阶段目标

### Part 1：ISA 语义与 data layout

修改：

- `isa.py`
- `ops/matmul.py`

目标：理解 LPU data layout、ISA 指令语义和 tiled matmul 的基本执行过程。

验证：

~~~bash
python3 tests/test_part1.py
~~~

### Part 2：TaskGen 与地址计算

修改：

- `ops/matmul_taskgen.py`

阅读：`task_gen.py`。

目标：理解 TaskGen 指令生成、L1/L0 分块和各级存储空间的地址计算。

验证：

~~~bash
python3 tests/test_part2.py
~~~

### Part 3：多执行单元同步

修改：

- `ops/matmul_sync.py`

参考：[set/wait 机制说明](https://tju-opentpu.feishu.cn/wiki/BXIiwUfGKiDFpjkskuYcY7D4nPh)。

目标：理解不同执行单元之间的依赖关系，以及 `set_flag` 和 `wait_flag` 的使用方法。

验证：

~~~bash
python3 tests/test_part3.py
~~~

## 环境

- Python 3.10+
- NumPy
- PyTorch，需要支持 `torch.bfloat16`
- Part 2/3 使用教程自带的 `lib/liblpu_func.so`

从 tutorial 根目录看，运行测试所需的动态库位于相对路径 `lib/liblpu_func.so`：

~~~text
./
├── lib/
│   └── liblpu_func.so
├── ops/
└── tests/
~~~


## 公共文件

- `common.py`：硬件常量、buffer 容量和地址对齐；
- `utils.py`：layout 变换、对齐和 bfloat16 编码；
- `semantic.py`：Broadcast、Binary、Unary、Reduce 原子语义；
- `task_gen.py`：legacy taskbin 指令编码；
- `tests/common_test.py`：Part 2/3 共用的测试工具；
- `tests/cmodel_runner.py`：通过教程内的 cmodel 动态库执行 taskbin。
