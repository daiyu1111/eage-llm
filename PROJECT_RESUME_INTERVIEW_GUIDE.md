# TinyChat W4A8 Matmul 项目面试复盘指南

这份文档用于解释简历中关于 TinyChatEngine W4A8 量化 `matmul` CPU 优化的描述。目标不是背术语，而是能在面试中把“为什么做、怎么做、代码在哪里、性能怎么来的、边界在哪里”讲清楚。

## 0. 简历原文

> 系统设计：基于 TinyChatEngine 完成 W4A8 量化 matmul CPU 优化，采用 int4 权重与 int8 动态激活路径，适配 block-wise scale / zero-point 反量化流程和 int4 packed 权重布局，降低模型存储占用与访存开销。实现并分析 reference、loop unrolling、pthread 多线程、x86 AVX2 SIMD 及组合优化版本；通过输出列展开提升激活数据复用，通过 pthread 按列并行提升多核利用率，通过 AVX2 intrinsic 完成 int4 解包、符号修正、int8 点积累加和 scale 融合。
>
> 效果：在单层 Linear 测试中验证正确性与性能，测试规模为 1x4096 激活与 32000x4096 int4 权重矩阵乘；在 NVIDIA A10 云实例的 x86 CPU AVX2 路径下，组合优化版本通过正确性校验，10 次平均耗时 23.57 ms，达到 11.12 GOPs。

## 0.1 建议改写后的简历版本

更准确的表述应该把“W4A8 路线是项目已有框架”与“你做的是 CPU kernel 优化”区分开：

> 系统设计：基于 TinyChatEngine 已有 W4A8 量化推理路径，完成 CPU 侧 int4-weight / int8-activation matmul kernel 优化，适配项目提供的 block-wise scale / zero-point 反量化流程和 int4 packed 权重布局，降低权重存储占用与访存开销。实现并分析 reference、loop unrolling、pthread 多线程、x86 AVX2 SIMD 及组合优化版本；通过输出列展开提升激活数据复用，通过 pthread 按输出列并行提升多核利用率，通过 AVX2 intrinsic 完成 int4 解包、符号修正、int8 点积累加和 scale 融合。
>
> 效果：在单层 Linear 测试中验证正确性与性能，测试规模为 1x4096 激活与 32000x4096 int4 packed 权重矩阵乘；在 NVIDIA A10 云实例的 x86 CPU AVX2 路径下，组合优化版本通过正确性校验，10 次平均耗时 23.57 ms，达到 11.12 GOPs。

如果简历空间更紧，可以压缩为：

> 基于 TinyChatEngine 已有 W4A8 量化路径，优化 CPU 侧 int4-weight / int8-activation matmul kernel，适配 block-wise scale / zero-point 反量化与 int4 packed 权重布局；实现 reference、loop unrolling、pthread 按列并行、x86 AVX2 SIMD 及组合优化版本，在 1x4096 x 32000x4096 单层 Linear 测试中通过正确性校验，10 次平均 23.57 ms，达到 11.12 GOPs。

面试中要主动澄清：

> W4A8 量化路线和基础数据格式是 TinyChatEngine 项目已有的，我的工作重点是理解并适配这条路径，在 CPU matmul kernel 层完成不同优化版本的实现、验证和性能分析。

## 1. 一分钟项目介绍

这是一个面向端侧/云实例 CPU 推理的 LLM Linear 层优化项目。项目基于 TinyChatEngine 的算子框架，重点优化低比特量化后的矩阵乘，也就是把 Linear 层中的 `A x W^T` 跑得更快。

我基于项目已有的 `W4A8` 路径做 CPU kernel 优化：权重离线量化为 `int4` 并 packed 存储，激活在运行时按 block 动态量化为 `int8`。kernel 里先按布局解包 int4 权重，把 `0~15` 的 nibble 修正到 `-8~7`，再和 int8 激活做整数点积，最后乘上 activation scale 和 weight scale 得到 float 输出。

优化过程分成几个版本：reference 版本保证正确性；loop unrolling 版本一次计算 4 个输出列来复用同一段激活；pthread 版本把输出列切给多个线程；SIMD 版本用 x86 AVX2 intrinsic 做批量解包和点积；最终 all techniques 版本把多线程、SIMD 和更深的 block 展开组合起来。在 `1x4096` 激活乘 `32000x4096` int4 权重的单层 Linear 测试中，组合优化版本在 x86 AVX2 路径下通过正确性校验，10 次平均 `23.57 ms`，约 `11.12 GOPs`。

## 2. 项目代码入口

| 主题 | 关键位置 | 面试时怎么讲 |
| --- | --- | --- |
| Linear int4 算子入口 | `transformer/include/ops/linear.h`、`transformer/src/ops/linear.cc` | `Linear_FP_int4` 负责加载 int4 权重、scale、offset、zero point，并在 `forward` 中组织 `matmul_params`。 |
| matmul 参数结构 | `kernels/matmul.h` | `matmul_params` 把 A/B/C 矩阵、权重 scale、activation scale、block size 等传给 kernel。 |
| 六个实验版本 | `kernels/starter_code/*.cc` | `IMP` 编译宏选择 reference、loop unrolling、multithreading、SIMD、组合优化等实现。 |
| 激活动态量化 | `kernels/quantizer.cc` | `quantize_fp32_to_int8` 对每 32 个 activation 求最大绝对值，生成 int8 和对应 scale。 |
| 测试脚本 | `transformer/evaluate.sh` | 通过 `make ... IMP=<id>` 编译不同实现，再运行 `test_linear`。 |
| 单层 Linear 测试 | `transformer/tests/test_linear.cc` | 固定测试 `m=1,n=32000,k=4096`，用 reference 输出做正确性对比。 |

## 3. 逐句解释简历内容

### 3.1 “基于 TinyChatEngine”

这句话说明项目不是孤立写一个函数，而是在 TinyChatEngine 的推理框架里接入 Linear 算子。

具体对应到代码：

- `Linear_FP_int4` 是 int4 权重 Linear 层封装。
- 构造函数加载 `weight_int4.bin`、`scaling_factor_int4.bin`、`offset_int4.bin`、`zero_point_int4.bin`。
- `Linear_FP_int4::forward` 把输入、权重、输出、scale 等信息填进 `matmul_params`。
- `IMP` 宏决定实际调用哪个 kernel：`mat_mul_reference`、`mat_mul_loop_unrolling`、`mat_mul_multithreading`、`mat_mul_simd_programming`、`mat_mul_multithreading_loop_unrolling` 或 `mat_mul_all_techniques`。

面试回答：

> 我不是只写了一个脱离系统的 matmul demo，而是沿着 TinyChatEngine 原有 Linear 算子接口接入。上层仍然调用 `Linear_FP_int4::forward`，底层通过 `matmul_params` 把矩阵指针、量化参数和 block size 传给不同 kernel，实现可以通过编译宏切换，方便做 correctness 和 performance 对比。

### 3.2 “完成 W4A8 量化 matmul CPU 优化”

这里要注意边界：`W4A8` 量化路线和基础数据格式来自 TinyChatEngine 项目框架，不建议说成自己原创设计。你的工作重点是基于这条已有路径完成 CPU `matmul` kernel 的实现、优化、验证和分析。

`W4A8` 的含义：

- `W4`：weight 用 4 bit 存储，也就是 `int4`。
- `A8`：activation 用 8 bit 计算，也就是 `int8`。

为什么这样设计：

- LLM 参数主要在权重里，权重压到 int4 可以显著减少模型存储和权重访存。
- 激活是运行时产生的，分布会随输入变化。把激活动态量化到 int8，比压到 int4 更稳，也更适合 AVX2/NEON 的整数向量计算。
- Linear 层本质是矩阵乘，重复度高，适合做低比特解包、整数点积和 SIMD 优化。

面试回答：

> W4A8 是 TinyChatEngine 项目已有的低比特推理路径，我的工作不是重新设计这条量化路线，而是在它的数据格式和反量化参数约束下优化 CPU matmul kernel。权重占模型存储大头，离线压到 int4 能降低带宽；激活运行时动态量化成 int8。kernel 里用 int8 激活和解包后的 int4 权重做整数点积，最后用 scale 还原成 float 输出。

### 3.3 “采用 int4 权重与 int8 动态激活路径”

权重路径：

- 权重离线量化并保存到 `weight_int4.bin`。
- 每两个 int4 权重 packed 到一个 `uint8_t` 中。
- 低 4 bit 保存一个权重，高 4 bit 保存另一个权重。

激活路径：

- 输入仍然以 float 进入 `Linear_FP_int4::forward`。
- kernel 开头调用 `quantize_fp32_to_int8`。
- 每 32 个 float activation 组成一个 block，计算该 block 的最大绝对值。
- scale 约等于 `max_abs / 127`，量化值约等于 `round(x / scale)`。
- 量化后的 int8 存入 `A.int8_data_ptr`，scale 存入 `A_scales`。

为什么叫“动态激活”：

- 权重是离线固定量化的。
- 激活每次 forward 都可能不同，所以每次运行时重新量化。

面试回答：

> 权重是静态的，提前量化并 packed 存储；激活是动态的，进入 Linear 时根据当前输入按 block 求 scale，再量化到 int8。这样既减少了权重访存，也保留了激活的动态范围。

### 3.4 “适配 block-wise scale / zero-point 反量化流程”

项目里不是整层只用一个 scale，而是 block-wise 量化。核心字段在 `matmul_params` 中：

- `scales`：权重每个 block 的 scale。
- `offset`：部分 int4 路径支持的 offset。
- `zero_point`：权重量化零点，常见是 `8`，用于把 `0~15` 转成以 0 为中心的范围。
- `A_scales`：激活每个 block 的动态 scale。
- `block_size`：当前实现主要假设为 `32`。

基本反量化关系：

```text
weight_int4_value = packed_nibble
weight_signed = weight_int4_value - zero_point
weight_dequant = weight_signed * weight_scale

activation_dequant = activation_int8 * activation_scale

output += sum(activation_int8 * weight_signed) * activation_scale * weight_scale
```

重要点：

- 优化版本不会先把完整权重反量化成 float 矩阵再乘。
- 它会把“整数点积”和“乘 scale”融合起来。
- 这样减少中间数据写回，避免失去低比特权重带来的带宽收益。

面试回答：

> block-wise 量化的好处是每 32 个元素有自己的 scale，可以更贴近局部数据分布。kernel 中先对 int8 activation 和解包后的 int4 weight 做整数累加，再乘 `A_scale * W_scale`。这相当于把反量化融合进 matmul，而不是先生成一份完整的 float 权重矩阵。

### 3.5 “int4 packed 权重布局”

int4 只有 4 bit，一个 byte 可以存两个权重：

```text
uint8_t packed = low_weight | (high_weight << 4)
low_weight  = packed & 0x0F
high_weight = packed >> 4
```

普通顺序布局可能是：

```text
(w0,w1), (w2,w3), (w4,w5), ...
```

为了 SIMD，x86 路径使用更适合 256-bit AVX2 load 的布局。代码注释中描述为：

```text
origin order: (w0,w1), (w2,w3), ... (w62,w63)
expected layout: (w0,w32), (w1,w33), ... (w31,w63)
```

这样一次 `_mm256_loadu_si256` 可以加载 32 个 byte，也就是 64 个 int4 权重。低 4 bit 对应前 32 个权重，高 4 bit 对应后 32 个权重，刚好分别和两段 32 个 int8 activation 做点积。

面试回答：

> int4 packed 的目标是让两个 4-bit 权重共用一个 byte，真实节省权重带宽。但 packed 后不能直接乘，需要运行时解包。为了让 AVX2 解包后能直接和 activation 对齐，项目对权重布局做了适配，例如 x86 下把 64 个权重组织成低半部分 32 个、高半部分 32 个，方便一条 256-bit load 后批量处理。

### 3.6 “降低模型存储占用与访存开销”

存储收益：

- FP32 权重：每个权重 4 byte。
- INT8 权重：每个权重 1 byte。
- INT4 packed 权重：每个权重 0.5 byte。

对 `32000 x 4096` 权重矩阵：

```text
权重数量 = 32000 * 4096 = 131,072,000
FP32 存储约 = 131,072,000 * 4 = 524,288,000 bytes ≈ 500 MiB
INT4 packed 存储约 = 131,072,000 / 2 = 65,536,000 bytes ≈ 62.5 MiB
```

即使考虑 scale/offset 的额外开销，权重主体访存仍大幅下降。

面试回答：

> Linear 层大矩阵乘往往很吃权重带宽。int4 packed 把权重主体从 FP32 的 4 byte 降到 0.5 byte，理论上是 1/8 的权重数据量。实际还有 scale 等元数据，但主体带宽下降非常明显。

### 3.7 “实现并分析 reference、loop unrolling、pthread 多线程、x86 AVX2 SIMD 及组合优化版本”

六个版本对应 `transformer/evaluate.sh`：

| IMP | 名称 | 文件 | 核心作用 |
| --- | --- | --- | --- |
| 0 | reference | `kernels/starter_code/reference.cc` | 标量实现，保证正确性基准。 |
| 1 | loop_unrolling | `kernels/starter_code/loop_unrolling.cc` | 一次计算 4 个输出列，复用 activation。 |
| 2 | multithreading | `kernels/starter_code/multithreading.cc` | pthread 按输出列切分，利用多核。 |
| 3 | simd_programming | `kernels/starter_code/simd_programming.cc` | 用 AVX2/NEON intrinsic 做向量化。 |
| 4 | multithreading_loop_unrolling | `kernels/starter_code/multithreading_loop_unrolling.cc` | 多线程 + 输出列展开。 |
| 5 | all_techniques | `kernels/starter_code/all_techniques.cc` | 多线程 + SIMD + 更深 block 展开。 |

面试回答：

> 我按逐层优化的方式实现和对比：reference 定义正确计算；loop unrolling 提高单线程数据复用；multithreading 提高多核利用；SIMD 提高单核向量吞吐；最后 all techniques 把这些优化组合起来，用同一套测试验证正确性和性能。

### 3.8 “通过输出列展开提升激活数据复用”

普通版本一次计算一个输出列：

```text
for col:
    load activation block
    load weight[col] block
    compute C[col]
```

输出列展开后一次计算 4 个输出列：

```text
for col += 4:
    load one activation block
    load weight[col + 0]
    load weight[col + 1]
    load weight[col + 2]
    load weight[col + 3]
    update acc0, acc1, acc2, acc3
```

收益：

- 同一段 activation 同时服务多个输出列。
- 减少循环控制开销。
- 增加寄存器中的累加器数量，提高指令级并行机会。

代价：

- 代码更长。
- 对输出列数对齐有要求，例如 `n % 4 == 0`。
- 展开过深会增加寄存器压力。

面试回答：

> 这里不是简单少几次循环跳转，更重要的是复用 activation。因为同一行输入要和很多输出列的权重做点积，一次展开 4 列后，同一个 activation block 可以同时更新 4 个累加器，减少重复读取和循环开销。

### 3.9 “通过 pthread 按列并行提升多核利用率”

输出矩阵 `C` 的不同列彼此独立：

```text
C[row][col0] 和 C[row][col1] 使用不同权重列
最终写入不同输出地址
```

因此可以按列区间切分：

```text
thread 0: col [0, n/4)
thread 1: col [n/4, n/2)
thread 2: col [n/2, 3n/4)
thread 3: col [3n/4, n)
```

在 `all_techniques` 中，线程数设为 `8`，每个线程处理一段输出列。因为每个线程写不同的 `C[row * n + col]` 范围，正常情况下没有写冲突。

收益：

- 把大量输出列分发到多个 CPU core。
- 对 `n=32000` 这种大输出维度很适合。

限制：

- 线程创建和 join 有开销。
- 线程数太多可能导致调度开销、cache 争用或内存带宽饱和。
- 按列切分会让所有线程共享同一行 activation，但读共享不会造成写冲突。

面试回答：

> 我按输出列切分线程，因为不同输出列独立，写回地址也不同，所以实现简单且没有加锁需求。这个测试里 `n=32000`，列数足够大，任务粒度能覆盖 pthread 创建和同步开销。

### 3.10 “通过 AVX2 intrinsic 完成 int4 解包、符号修正、int8 点积累加和 scale 融合”

x86 AVX2 路径核心步骤在 `simd_programming.cc` 和 `all_techniques.cc`。

1. 加载 packed int4 权重：

```cpp
__m256i raw_w = _mm256_loadu_si256(w_start);
```

2. 解出低 4 bit 和高 4 bit：

```cpp
lower = raw_w & 0x0F
upper = (raw_w >> 4) & 0x0F
```

3. 用 zero point 修正到 signed 范围：

```cpp
w_0   = lower - 8
w_128 = upper - 8
```

4. 适配 `_mm256_maddubs_epi16` 的 unsigned/signed 输入要求：

```cpp
abs_w = abs(w)
signed_activation = sign(activation, w)
```

原因是 `_mm256_maddubs_epi16` 把第一个输入当 unsigned byte，第二个输入当 signed byte。权重本来可能有负数，因此把符号转移到 activation 上，权重取绝对值。

5. 做 int8 点积：

```cpp
dot = _mm256_maddubs_epi16(abs_w, signed_activation)
sum32 = _mm256_madd_epi16(ones, dot)
float_sum = _mm256_cvtepi32_ps(sum32)
```

6. scale 融合：

```cpp
accumulator = fmadd(float_sum, A_scale * W_scale, accumulator)
```

面试回答：

> AVX2 版本的难点不是只调用 intrinsic，而是要适配指令的数据语义。权重 packed 后先用 mask 和 shift 解出两个 nibble，再减 8 得到 signed int4。由于 `_mm256_maddubs_epi16` 的第一个输入是 unsigned byte，我把权重取绝对值，把符号转移到 activation 上，然后做批量乘加。整数累加转 float 后直接乘 `A_scale * W_scale`，避免单独生成反量化权重。

## 4. 从一次 forward 看完整数据流

### 4.1 测试输入

`transformer/tests/test_linear.cc` 固定测试规模：

```cpp
const int m = 1, n = 32000, k = 4096;
```

含义：

- activation：`1 x 4096`。
- weight：`32000 x 4096`，以 int4 packed 形式存储为 `32000 x 2048` byte。
- output：`1 x 32000`。

### 4.2 加载权重和量化参数

`Linear_FP_int4` 构造函数加载：

- `weight_int4.bin`：packed int4 权重。
- `scaling_factor_int4.bin`：每个 block 的权重 scale。
- `offset_int4.bin`：部分反量化路径使用的 offset。
- `zero_point_int4.bin`：权重 zero point。

### 4.3 forward 组织参数

`Linear_FP_int4::forward` 设置：

- `params.A.data_ptr`：float activation。
- `params.A.int8_data_ptr`：动态量化后的 int8 activation 缓冲区。
- `params.A_scales`：activation block scale。
- `params.B.int4_data_ptr`：packed int4 权重。
- `params.scales`：权重 block scale。
- `params.offset`：权重 offset。
- `params.block_size = QK`，当前主要是 `32`。

### 4.4 激活动态量化

每个 starter kernel 开头调用：

```cpp
quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales,
                      A->row * A->column, block_size);
```

x86 版本一次处理 32 个 float：

- 用 AVX load 读取 4 个 `__m256`，共 32 个 float。
- 求最大绝对值。
- 计算 scale。
- 乘 `127 / max_abs`。
- round 到整数。
- pack 成 int8。

### 4.5 kernel 计算

每个输出元素本质是：

```text
C[row][col] = sum_k A[row][k] * W[col][k]
```

W4A8 kernel 中变为：

```text
int_sum = sum_k A_int8[row][k] * W_int4_signed[col][k]
C[row][col] += int_sum * A_scale[block] * W_scale[col][block]
```

### 4.6 正确性验证

`test_linear.cc` 中：

- `forward_ref(hidden_states, outputQ)` 生成 reference 输出。
- `forward(hidden_states, outputQ_fast)` 生成当前 `IMP` 实现输出。
- `check_two_equal(..., 1e-3)` 检查误差是否在阈值内。

面试回答：

> 我用同一个输入和同一份 int4 权重，先跑 reference 得到基准输出，再跑优化版本，然后用 `1e-3` 阈值比较。这样能保证优化没有改变数值语义。

## 5. 每个实现版本怎么讲

### 5.1 reference

做什么：

- 最直观地遍历 `row -> col -> block -> qj`。
- 解包 int4。
- 减 zero point。
- 和 int8 activation 做标量乘加。
- 乘 scale 得到 float 输出。

价值：

- 定义正确答案。
- 方便后续优化版本对齐。
- 便于理解 W4A8 数据流。

一句话：

> reference 版本不是为了快，而是为了定义正确性基准，后续所有优化都要保证和它输出一致。

### 5.2 loop unrolling

做什么：

- `col += 4`。
- 同时维护 `acc0~acc3`。
- 同一段 activation 与 4 个输出列的权重分别相乘。

价值：

- activation 数据复用。
- 减少循环控制开销。
- 增加单线程内的并行度。

一句话：

> loop unrolling 的核心收益是让同一段 activation 同时服务多个输出列，而不是每个输出列都重复走一遍相同的输入读取逻辑。

### 5.3 pthread multithreading

做什么：

- 每个线程拿一个输出列区间。
- 线程内部仍然按 reference 逻辑计算。
- 主线程创建 pthread 后 join。

价值：

- 利用多核 CPU。
- 对大 `n` 的 Linear 层效果更明显。

一句话：

> 多线程版本利用的是输出列之间的独立性，每个线程写不同列区间，因此不需要锁。

### 5.4 SIMD programming

做什么：

- 用 AVX2 一次加载 256 bit packed 权重。
- 批量解出低半和高半 int4。
- 批量修正 zero point。
- 用 `_mm256_maddubs_epi16` 和 `_mm256_madd_epi16` 完成 int8 点积累加。
- 用 `_mm256_fmadd_ps` 融合 scale。

价值：

- 提高单核吞吐。
- 把标量的逐元素解包、乘加改成向量批量操作。

一句话：

> SIMD 版本把最核心的 inner loop 从逐元素计算变成一批元素一起计算，是 CPU 单核性能提升的关键。

### 5.5 multithreading + loop unrolling

做什么：

- 线程间按列切分。
- 线程内一次计算 4 列。

价值：

- 同时利用多核并行和单线程 activation 复用。

一句话：

> 这个版本把线程级并行和列展开带来的数据复用叠加起来，但还没有用 AVX2 做最深层的向量化。

### 5.6 all techniques

做什么：

- 使用 pthread 按列并行。
- x86 下用 AVX2 解包和点积。
- 每轮处理 4 个 block，也就是 128 个 int4 权重。
- 直接累加带 scale 的 float 部分和。

价值：

- 权重低比特降低带宽。
- 多线程提高多核利用率。
- SIMD 提高单核向量吞吐。
- block 展开减少循环和指针更新开销。

一句话：

> all techniques 快不是因为某一个技巧特别神，而是把带宽、线程并行、SIMD 吞吐和循环开销几个瓶颈一起压低了。

## 6. 性能指标怎么解释

### 6.1 理论计算量

测试规模：

```text
m = 1
n = 32000
k = 4096
```

矩阵乘每个输出元素需要 `k` 次乘法和 `k` 次加法，通常按 `2 * m * n * k` 记 ops：

```text
ops = 2 * 1 * 32000 * 4096
    = 262,144,000 ops
    = 262.144 M ops
```

### 6.2 GOPs 计算

实测 10 次平均耗时：

```text
time = 23.57 ms = 0.02357 s
GOPs = 262,144,000 / 0.02357 / 1e9
     ≈ 11.12 GOPs
```

### 6.3 如何诚实表达

可以说：

> 在单层 Linear 测试中，输入为 `1x4096`，权重为 `32000x4096` 的 int4 packed 矩阵。组合优化版本在 A10 云实例的 x86 CPU AVX2 路径下通过 reference 正确性校验，10 次平均耗时 `23.57 ms`，按 `2mnk` 计算约 `11.12 GOPs`。

不要说：

- “整个 LLaMA 模型端到端达到 11.12 GOPs。”
- “所有平台都能达到 23.57 ms。”
- “这是 GPU 性能。”

边界：

- 这是单层 Linear。
- 这是 x86 CPU AVX2 路径。
- 这是特定机器、特定输入规模、特定实现的结果。

### 6.4 11.12 GOPs 是不是很慢？

如果和现代 CPU 的理论峰值、MKL/OpenBLAS 这类高度优化 GEMM 或纯 int8 GEMM benchmark 比，`11.12 GOPs` 确实不算高。但这个数字不能直接按“裸 GEMM 峰值”理解，因为这个 kernel 做的不是理想的 `int8 x int8 -> int32` 大矩阵乘。

这个测试包含了额外工作：

- float activation 运行时动态量化到 int8。
- int4 packed 权重加载。
- int4 低/高 4 bit 解包。
- zero point 修正，把 `0~15` 转为 `-8~7`。
- 为适配 `_mm256_maddubs_epi16` 做符号处理。
- int8 点积累加后转 float。
- 融合 `A_scale * W_scale` 做反量化。
- 写回 float output。

测试形状也不利于跑出大 GEMM 峰值：

```text
A: 1 x 4096
W: 32000 x 4096
C: 1 x 32000
```

这更接近 batch size 为 1 的 matrix-vector / skinny GEMM。它不像大 batch GEMM 那样能让同一块权重服务很多行 activation，因此更容易受到权重访存、cache 行为、解包开销和线程调度影响。

面试回答：

> 是的，如果和成熟 BLAS 或纯 int8 GEMM 峰值比，11.12 GOPs 不高。但这个结果对应的是 batch=1 的 W4A8 Linear kernel，里面包含 activation 动态量化、int4 packed 权重解包、zero-point 修正、符号处理、scale 融合和 float 输出写回，不是理想的裸 GEMM。这个项目重点是理解并实现低比特 matmul 的 CPU 优化路径，并通过不同版本验证优化收益，而不是声称超过成熟库。

简历中建议只把它作为事实指标，不要包装成“高吞吐突破”。如果后续能补充各版本耗时，最好改成“相对加速比 + 最终耗时”，例如：

```text
reference: xx ms
loop unrolling: xx ms
pthread: xx ms
AVX2 SIMD: xx ms
all techniques: 23.57 ms
speedup: x.xx 倍
```

### 6.5 实际 A10 云实例 CPU 硬件参数

注意：`NVIDIA A10` 指的是 GPU 型号，不是 CPU 型号。这个项目的性能数字是在 A10 云实例里的 **x86 CPU 路径** 上测到的，真正影响 CPU kernel 性能的是云实例分配到的 CPU、vCPU 数、缓存、指令集和虚拟化环境。

根据实际 `lscpu` 输出，本次测试 CPU 环境为：

| 项目 | 参数 |
| --- | --- |
| 架构 | `x86_64` |
| CPU 型号 | `Intel(R) Xeon(R) Platinum 8369B CPU @ 2.70GHz` |
| vCPU 数 | `8` |
| 物理/虚拟拓扑 | `1 socket, 4 cores/socket, 2 threads/core` |
| 虚拟化 | `KVM` 完全虚拟化 |
| L1d Cache | `192 KiB`，4 instances |
| L1i Cache | `128 KiB`，4 instances |
| L2 Cache | `5 MiB`，4 instances |
| L3 Cache | `48 MiB`，1 instance |
| NUMA | `1` 个 NUMA 节点，CPU `0-7` |
| 关键 SIMD 指令集 | `AVX`, `AVX2`, `FMA`，同时 CPU flags 中也支持多种 `AVX512` 扩展 |

虽然这颗 CPU 支持 AVX512，但本项目 Makefile 在 x86_64 下使用的是：

```text
-mavx2 -mfma -DQM_x86
```

因此简历和面试中应表述为 **x86 CPU AVX2 路径**，不要写成 AVX512 优化，也不要把 `23.57 ms / 11.12 GOPs` 说成 A10 GPU 结果。

推荐写法：

> 测试环境为 NVIDIA A10 云实例中的 x86 CPU 路径，CPU 为 Intel Xeon Platinum 8369B @ 2.70GHz，容器可见 8 vCPU，支持 AVX2/FMA；本项目编译参数启用 AVX2/FMA，组合优化版本在单层 Linear 测试中 10 次平均耗时 23.57 ms，对应 11.12 GOPs。

如果简历空间有限，可以不写完整硬件参数，只在项目报告或面试材料里保留。简历里写：

> 在 A10 云实例的 x86 CPU AVX2 路径下测试。

面试中再补充：

> A10 是 GPU 型号，但我的 kernel 跑的是 CPU 路径。当时容器可见 8 个 vCPU，CPU 型号是 Intel Xeon Platinum 8369B，支持 AVX2/FMA。虽然机器也有 AVX512 flags，但我的实现和编译参数走的是 AVX2。

## 7. 高频面试问答

### Q0：W4A8 路线是你自己设计的吗？

不是。更准确地说，W4A8 量化路线、int4 packed 权重格式、block-wise scale 等基础框架来自 TinyChatEngine 项目。我做的是在这个已有数据路径下，理解并适配它的量化参数和权重布局，然后实现、补全和分析 CPU matmul kernel 的多种优化版本，包括 reference、loop unrolling、pthread、AVX2 SIMD 和 all techniques。

### Q1：为什么选 W4A8，而不是全 FP32？

全 FP32 权重带宽和存储压力太大。对于大模型 Linear 层，权重矩阵很大，很多时候性能受访存限制。W4A8 把权重主体压到 4 bit，显著降低权重读取量；激活保留 int8，精度和计算便利性相对平衡。

### Q2：为什么不是 W8A8？

W8A8 更简单，精度也通常更稳，但权重存储是 int4 的 2 倍。项目重点是低比特权重压缩和访存优化，所以选择 W4A8 来进一步降低权重带宽。

### Q3：为什么不是 W4A16？

W4A16 可以避免 activation int8 量化误差，但计算路径更偏 float/half，难以充分利用 int8 点积指令。W4A8 可以把 activation 和 weight 都转到整数点积路径，适合 SIMD 批量计算。

### Q4：block-wise 量化比 per-tensor 量化好在哪里？

per-tensor 一整层共享一个 scale，如果局部数值分布差异大，小值区域会损失精度。block-wise 每 32 个元素一组，有自己的 scale，可以更好适配局部分布，在压缩率和误差之间折中。

### Q5：int4 权重为什么要 packed？

如果 int4 仍然用一个 byte 存一个权重，就浪费了 4 bit，带宽收益只有 int8 水平。packed 后两个权重放一个 byte，才能真正把权重主体压到 0.5 byte/weight。

### Q6：packed 后有什么代价？

代价是 kernel 里要解包。每次计算前需要用 mask 取低 4 bit，用 shift 取高 4 bit，再做 zero point 修正。布局如果不适合 SIMD，还会有额外 shuffle 或重排开销。

### Q7：为什么 Linear/matmul 是优化重点？

Transformer 里大量计算都在 Linear 层，尤其 attention projection、MLP projection、lm_head 都是矩阵乘。Linear 层权重大、调用频繁、数据访问量大，因此优化 matmul 对推理性能影响很直接。

### Q8：reference 版本怎么保证正确性？

reference 版本按最直接的数学流程实现：逐 block 解包 int4，减 zero point，和 int8 activation 做乘加，再乘 activation scale 和 weight scale。它逻辑简单，作为优化版本的数值基准。

### Q9：loop unrolling 为什么按输出列展开？

因为同一行 activation 会和很多输出列的权重相乘。按输出列展开后，同一段 activation 可以同时更新多个输出列的累加器，提升数据复用，减少重复读 activation 和循环控制开销。

### Q10：pthread 为什么按列切分？是否有写冲突？

输出列之间独立，每个线程负责不同列区间，写入 `C[row][col]` 的地址不同，所以没有写冲突。所有线程共享读取 activation 和权重 scale，但读共享不需要加锁。

### Q11：为什么线程数不是越多越好？

线程多会带来创建、调度、同步开销，也可能导致 cache 争用和内存带宽饱和。当任务粒度不够大或者带宽已经满了，继续加线程可能不再提升，甚至变慢。

### Q12：AVX2 如何完成 int4 解包？

先用 `_mm256_loadu_si256` 读取 packed 权重。低 4 bit 用 `_mm256_and_si256(raw, 0x0F)` 取出，高 4 bit 用 `_mm256_srli_epi16(raw, 4)` 后再 mask。然后用 `_mm256_sub_epi8(..., 8)` 把 `0~15` 修正为 `-8~7`。

### Q13：`_mm256_maddubs_epi16` 为什么需要符号修正？

这条指令把第一个输入解释为 unsigned byte，把第二个输入解释为 signed byte。但解包后的权重是 signed 值，不能直接作为第一个输入。项目中做法是取权重绝对值作为 unsigned 输入，再根据权重符号调整 activation 的符号，从而等价完成 signed 乘法。

### Q14：scale 融合发生在哪一步？

整数点积先得到 int32 部分和，再转成 float，然后乘 `A_scale * W_scale` 并累加到输出 accumulator。这样不需要先把整块权重反量化成 float 数组，减少中间内存和访存开销。

### Q15：为什么不先完整反量化权重再做 FP32 matmul？

那样会把 int4 权重重新膨胀成 float，失去低比特存储和带宽收益，还会产生额外中间内存写入。融合式做法只在寄存器里解包和乘 scale，更符合低比特 kernel 的目的。

### Q16：23.57 ms 和 11.12 GOPs 怎么算？

计算量按 `2mnk`：

```text
2 * 1 * 32000 * 4096 = 262,144,000 ops
```

耗时 `23.57 ms = 0.02357 s`：

```text
262,144,000 / 0.02357 / 1e9 ≈ 11.12 GOPs
```

### Q17：为什么组合优化不等于各优化收益简单相乘？

因为不同优化可能竞争同一个瓶颈。例如 SIMD 提高计算吞吐后，瓶颈可能转向内存带宽；线程增多后也可能更快触及带宽上限。组合优化通常有叠加收益，但不会线性相乘。

### Q18：这个 kernel 更偏访存瓶颈还是计算瓶颈？

低比特权重的初衷是缓解访存瓶颈，但引入了解包、符号处理和 scale 融合等额外计算。优化后瓶颈可能在访存和计算之间切换。可以通过改变线程数、观察 SIMD 提升幅度、用 perf 看 cache miss/IPC/带宽指标来判断。

### Q19：你有没有做 CUDA/Metal？

稳妥回答：

> 仓库里有 CUDA、Metal、NEON 等后端，我阅读过它们在整体系统中的位置。但我简历这段重点是 CPU W4A8 matmul kernel，实际深入实现和分析的是 reference、loop unrolling、pthread、x86 AVX2 SIMD 和组合优化路径。

### Q20：如果让你继续优化，会做什么？

可以从四个方向回答：

- 线程池复用，避免每次 forward 创建和销毁 pthread。
- 根据 CPU core 数和带宽情况调优线程数。
- 尝试更好的列分块，让 activation cache 复用和权重连续访问更平衡。
- 用 perf/VTune 分析 cache miss、IPC、指令热点，再决定是优化解包、scale 加载还是线程调度。

## 8. 面试中的边界表述

推荐说法：

> 我这段项目的核心工作是 CPU 侧 W4A8 量化 matmul kernel 优化。项目整体包含多种后端，但我的简历数字和实现分析主要对应 x86 CPU AVX2 路径下的单层 Linear 测试。我用 reference 输出做正确性基准，用 `m=1,n=32000,k=4096` 的固定规模统计 10 次平均耗时，并按 `2mnk` 计算 GOPs。

避免说法：

- “我完整重写了 TinyChatEngine。”
- “我实现了完整 LLaMA 端到端推理加速。”
- “A10 GPU 上跑到了 11.12 GOPs。”
- “所有优化在所有机器上都会提升。”

## 9. 三分钟展开版口述稿

这个项目主要是做 TinyChatEngine 中低比特 Linear 层的 CPU kernel 优化。背景是 LLM 推理中 Linear 层权重矩阵很大，访存压力明显，尤其在端侧或普通 CPU 上，如果直接用 FP32 权重，模型存储和权重读取都会比较重。项目本身已经提供 W4A8 量化路径，我的工作是在这条路径下优化 CPU matmul kernel，适配 int4 packed 权重和运行时 int8 activation。

在数据流上，权重加载时带有 block-wise 的 scale、offset 和 zero point。activation 每次 forward 时按 32 个元素一组动态量化，生成 int8 activation 和对应的 activation scale。kernel 内部不会先完整反量化权重，而是边解包 int4、边做 int8 整数点积，最后把整数累加结果乘上 `A_scale * W_scale` 得到 float 输出。

优化上我做了几个逐层版本。reference 版本用于定义正确性；loop unrolling 版本一次计算 4 个输出列，提升 activation 数据复用；pthread 版本按输出列切分任务，利用多核 CPU；SIMD 版本使用 AVX2 intrinsic 完成 int4 解包、zero point 修正、符号处理、int8 点积和 scale 融合；最终 all techniques 版本把多线程、SIMD 和 block 展开组合起来。

测试上，我使用单层 Linear，输入是 `1x4096`，权重是 `32000x4096` 的 int4 packed 矩阵。优化版本输出和 reference 输出用 `1e-3` 阈值比较，保证正确性。性能方面，在 A10 云实例的 x86 CPU AVX2 路径下，组合优化版本 10 次平均耗时 `23.57 ms`。按 `2 * 1 * 32000 * 4096` 计算，理论操作量是 `262.144M ops`，对应约 `11.12 GOPs`。

## 10. 最后速记

- W4A8：int4 weight + int8 activation。
- 权重离线量化，激活动态量化。
- block size 主要是 32。
- int4 packed：两个权重一个 byte。
- zero point 常见为 8，把 `0~15` 转成 `-8~7`。
- reference 保正确性。
- loop unrolling 复用 activation。
- pthread 按输出列并行，无写冲突。
- AVX2 负责批量解包、符号修正、点积和 scale 融合。
- all techniques 组合了带宽优化、多核并行和单核 SIMD。
- 测试是单层 Linear，不是完整模型端到端。
- `23.57 ms` 和 `11.12 GOPs` 来自 `2mnk / time`。
