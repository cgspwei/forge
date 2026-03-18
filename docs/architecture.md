# Forge 架构设计文档

> **版本**: v0.1
> **状态**: 草案

---

## 1. 项目目标

### 1.1 我们要解决什么问题

现代深度学习框架（PyTorch 2.x、JAX、XLA）内部都收敛到了同一个架构模式：

```
eager 前端 → 图捕获 → 中间表示 → 编译优化 → 后端执行
```

但这些系统的代码规模巨大（PyTorch ~300 万行、XLA ~50 万行），大量工程细节掩盖了核心设计思想。一个初学者想要理解"张量运算是怎么表示和执行的"、"自动微分是怎么工作的"、"编译器是怎么优化计算图的"，几乎无从下手。

**Forge 的目标是：用最小规模的现代 C++ 代码（~6000 行），完整实现这条 pipeline，让每个环节都可读、可调试、可修改。**

### 1.2 不是什么

Forge **不是**：
- 生产级 ML 框架（不追求性能极致）
- GPU/CUDA 系统（仅 CPU 后端）
- 分布式训练系统
- PyTorch/TensorFlow 的替代品

### 1.3 成功标准

| 标准 | 具体目标 |
|------|----------|
| **完整性** | 能跑通 MLP 训练和 tiny transformer 推理 |
| **可读性** | 每一层代码量 < 2000 行，无"魔法"依赖 |
| **可对照性** | 每个设计决策都能映射到 PyTorch/XLA 的对应概念 |
| **可扩展性** | 添加一个新算子不需要修改超过 3 个文件 |

---

## 2. 为什么要分层

### 2.1 从一个简单例子说起

用户写了这样的代码：

```cpp
Tensor x = randn({32, 128}, requires_grad=true);
Tensor w = randn({128, 64}, requires_grad=true);
Tensor y = relu(matmul(x, w));
Tensor loss = mean(y);
loss.backward();
```

这 5 行代码背后，系统需要做哪些事？

1. **表示张量**：`x` 是一个 32×128 的浮点数矩阵，要管理它的内存、形状、数据类型
2. **执行运算**：`matmul`、`relu`、`mean` 各自怎么计算
3. **记录历史**：`backward()` 需要知道 `loss` 是怎么从 `x`、`w` 一步步算出来的
4. **计算梯度**：沿着记录的历史，反向逐步求出 `x.grad` 和 `w.grad`

如果进一步，用户想编译优化：

```cpp
auto compiled = forge::compile([](Tensor x, Tensor w) {
    return relu(matmul(x, w));
});
Tensor out = compiled.run(x, w);
```

还需要：

5. **捕获计算图**：把 eager 执行的过程"录制"成一个静态图
6. **表示计算图**：用一种结构化的中间表示（IR）来描述这个图
7. **优化计算图**：常量折叠、死代码消除、算子融合
8. **执行计算图**：调度到 CPU 后端，高效执行

### 2.2 自然的职责划分

上面的需求自然地形成了以下分层：

```
需求                        对应层
─────────────────────────────────────
表示张量、管理内存           → Tensor Core
执行基本运算                 → Ops
记录历史、计算梯度           → Autograd
捕获计算图                   → Graph Capture
结构化表示计算图             → Graph IR
优化计算图                   → Compiler
调度执行                     → Runtime
```

### 2.3 为什么不能合并

**为什么 Tensor 和 Autograd 要分开？**

不是所有张量运算都需要梯度。推理阶段、数据预处理、常量计算——这些场景不需要 autograd 的开销。分开后，`TensorImpl` 只在 `requires_grad=true` 时才分配 autograd 元数据，实现"不用就不付费"。

PyTorch 也是这么做的：`TensorImpl` 中 autograd 信息是可选挂载的 `AutogradMeta`。

**为什么 Graph Capture 和 Graph IR 要分开？**

捕获（怎么把 eager 代码变成图）和表示（图长什么样）是两个独立关注点：

- Graph Capture 关心的是：怎么拦截算子调用、怎么处理代理张量、追踪的边界在哪里
- Graph IR 关心的是：节点怎么表示、边怎么连接、怎么遍历和变换

分开后，可以独立替换捕获策略（tracing vs symbolic），而不影响 IR 的设计。XLA 用 LazyTensor 捕获但用 HLO 作为 IR；PyTorch 用 Dynamo 捕获但用 FX Graph 作为 IR——捕获和表示是正交的。

**为什么 Graph IR 和 Compiler 要分开？**

IR 是数据，Compiler 是变换。IR 定义了图的结构和约束（SSA、类型规则），Compiler 是在这个结构上运行的一系列 pass。同一个 IR 可以接受不同的 pass 组合；同一个 pass 的逻辑也不应与特定 IR 实现耦合。

**为什么 Compiler 和 Runtime 要分开？**

编译是"离线"的图到图变换（可以慢），执行是"在线"的实际计算（必须快）。编译器不需要知道线程池怎么调度，Runtime 不需要知道常量折叠怎么做。

---

## 3. 分层架构

### 3.1 全局视图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户代码                               │
│  Tensor x = randn({32, 128});                               │
│  Tensor y = relu(matmul(x, w));                             │
│  loss.backward();                                           │
│  auto compiled = forge::compile(fn);                        │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │   Layer 1: Tensor Core  │  张量的表示与内存管理
    │   Shape, Storage, View  │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │   Layer 2: Ops          │  算子的计算实现
    │   unary, binary, reduce │
    └────────┬────────┬───────┘
             │        │
    ┌────────▼───┐ ┌──▼──────────────┐
    │  Layer 3:  │ │  Layer 4:       │
    │  Autograd  │ │  Graph Capture  │
    │  动态微分   │ │  追踪 eager→图  │
    └────────────┘ └──┬──────────────┘
                      │
          ┌───────────▼───────────┐
          │  Layer 5: Graph IR    │  计算图的中间表示
          │  Node, Edge, SSA DAG  │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Layer 6: Compiler    │  图优化 pass
          │  fold, DCE, simplify  │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Layer 7: Runtime     │  执行调度
          │  executor, threadpool │
          └───────────────────────┘
```

### 3.2 依赖规则

**严格单向依赖，下层不知道上层的存在。**

```
Tensor Core ← Ops ← Autograd
                  ← Graph Capture ← Graph IR ← Compiler ← Runtime
```

具体地：

| 层 | 可以依赖 | 不可以依赖 |
|----|---------|-----------|
| Tensor Core | 无 | 其他所有层 |
| Ops | Tensor Core | Autograd, Graph, Compiler, Runtime |
| Autograd | Tensor Core, Ops | Graph, Compiler, Runtime |
| Graph Capture | Tensor Core, Ops | Autograd, Compiler, Runtime |
| Graph IR | Tensor Core (仅类型定义) | Ops, Autograd, Capture, Compiler, Runtime |
| Compiler | Graph IR | Tensor Core, Ops, Autograd, Capture, Runtime |
| Runtime | Graph IR, Ops | Tensor Core, Autograd, Capture, Compiler |

注意 Autograd 和 Graph Capture 之间**没有**依赖关系。它们是两条独立的路径：
- Autograd 服务于 eager 模式下的梯度计算
- Graph Capture 服务于编译模式下的图追踪

### 3.3 两条执行路径

```
路径 A: Eager 执行 + Autograd (训练)
─────────────────────────────────────
用户代码 → Ops (前向计算 + 记录 GradFn)
         → Autograd Engine (反向传播)

路径 B: 编译执行 (推理/优化训练)
─────────────────────────────────────
用户代码 → Graph Capture (追踪)
         → Graph IR (构建)
         → Compiler (优化)
         → Runtime (执行)
```

---

## 4. 各层职责

### 4.1 Layer 1: Tensor Core

**一句话**：定义张量是什么——形状、数据类型、内存布局、存储管理。

**核心概念**：
- **Storage**：一块原始内存，不关心形状和步幅
- **View**：同一块 Storage 上的不同"视角"（形状 + 步幅 + 偏移）
- **Tensor**：用户拿到的句柄，轻量级，可自由拷贝（引用语义）

**关键设计**：存储与视图分离。`transpose()` 不拷贝数据，只改变步幅。多个 Tensor 可以共享同一块 Storage。

**业界对照**：
| Forge | PyTorch | 说明 |
|-------|---------|------|
| `Storage` | `StorageImpl` | 原始内存块 |
| `TensorImpl` | `TensorImpl` | 带元数据的存储视图 |
| `Tensor` | `Tensor` | 用户句柄（引用计数） |

### 4.2 Layer 2: Ops

**一句话**：定义张量上可以做什么运算——加减乘除、矩阵乘、激活函数、归约。

**核心概念**：
- **逐元素运算**：`add`, `mul`, `relu`, `sigmoid` 等，支持广播
- **归约运算**：`sum`, `mean`，沿指定维度收缩
- **矩阵运算**：`matmul`
- **形状运算**：`reshape`, `transpose`, `slice`

**关键设计**：Op 是纯函数，输入 Tensor 输出 Tensor，自身无状态。广播规则与 NumPy/PyTorch 一致。

**为什么独立成层**：Op 的计算逻辑被 Autograd（前向执行时调用）和 Runtime（编译执行时调用）共享。如果 Op 代码散落在 Autograd 或 Runtime 中，会造成重复。

### 4.3 Layer 3: Autograd

**一句话**：在 eager 执行时记录"怎么算的"，然后反向回溯计算梯度。

**核心概念**：
- **GradFn**：反向函数节点，每次前向运算时动态创建
- **Edge**：连接 GradFn 的有向边，构成反向计算图
- **Engine**：拓扑排序 + 逆序遍历，驱动整个反向传播

**关键设计**：动态计算图（define-by-run）。每次前向执行都构建新的图，`backward()` 后图即释放。

**业界对照**：
| Forge | PyTorch | JAX |
|-------|---------|-----|
| 动态图 + GradFn | 动态图 + Node | 函数变换 (jax.grad) |
| Engine 单线程 | Engine 多线程 | N/A（无命令式引擎） |
| 不支持高阶导数 | 支持高阶导数 | 原生支持 |

### 4.4 Layer 4: Graph Capture

**一句话**：把一段 eager 代码"录像"成一个静态计算图。

**核心概念**：
- **TracingContext**：追踪上下文，拦截所有 Op 调用
- **ProxyTensor**：不含真实数据的占位张量，只携带形状信息
- **compile(fn)**：入口函数，触发追踪过程

**关键设计**：基于追踪（tracing-based）的捕获。在追踪模式下，每个 Op 不真正计算，而是往 GraphBuilder 里添加一个节点。

**追踪的本质限制**：
- 数据依赖的控制流（`if tensor.item() > 0`）无法捕获
- 每组输入形状需要独立追踪

**业界对照**：
| Forge | PyTorch | XLA | JAX |
|-------|---------|-----|-----|
| 纯 tracing | Dynamo (字节码分析) | LazyTensor (惰性累积) | Jaxpr tracing |
| 无 graph break | 有 graph break + guard | 触发同步 | 禁止副作用 |
| ~500 行 | ~50K 行 | ~10K 行 | ~5K 行 |

### 4.5 Layer 5: Graph IR

**一句话**：计算图的结构化表示——节点是什么、边怎么连、怎么遍历和变换。

**核心概念**：
- **Node**：图中的一个计算节点（Op + 输入 + 输出形状）
- **Graph**：节点的有序集合，带有输入/输出标记
- **SSA 性质**：每个 NodeId 只定义一次，不可重赋值

**关键设计**：轻量级 SSA DAG。节点粒度与 eager Op 对齐（不像 HLO 那样低级），便于从 Graph Capture 直接生成。

**业界对照**：
| Forge | PyTorch FX | XLA HLO | StableHLO |
|-------|-----------|---------|-----------|
| 自定义 Graph | FX Graph | HloModule | MLIR Dialect |
| Op 级粒度 | Python 级粒度 | 硬件级粒度 | 标准化 HLO |
| ~800 行 | ~5K 行 | ~50K 行 | 基于 MLIR |

### 4.6 Layer 6: Compiler（本文档范围外，后续补充）

**一句话**：对 Graph IR 运行一系列优化 pass，产出等价但更高效的图。

计划实现的 pass：
1. **常量折叠** — 编译期计算已知结果
2. **死代码消除** — 移除不影响输出的节点
3. **代数简化** — `x * 1 → x`, `x + 0 → x`
4. **算子融合**（可选）— 合并可一次计算的相邻算子

### 4.7 Layer 7: Runtime（本文档范围外，后续补充）

**一句话**：按照拓扑顺序调度图中的节点到 CPU 后端执行。

计划实现：
- 按拓扑序逐节点执行
- 简单线程池实现并行
- 基本内存复用（buffer 重用）

---

## 5. 层间交互

### 5.1 关键接口

各层之间通过少量明确的接口类型交互：

```
Tensor Core ──(Tensor)──→ Ops
                            │
                     (Tensor + GradFn) ──→ Autograd
                     (Tensor + NodeId) ──→ Graph Capture
                                              │
                                        (Graph) ──→ Graph IR
                                                      │
                                               (Graph&) ──→ Compiler
                                                              │
                                                        (Graph) ──→ Runtime
```

- **Tensor**：Layer 1-4 之间传递的核心数据类型
- **Graph**：Layer 4-7 之间传递的核心数据类型
- Autograd 通过 `TensorImpl` 上的可选字段 `AutogradMeta` 挂载梯度信息——这是唯一的"跨层侵入"，但通过惰性分配保持了"不用就不付费"

### 5.2 Eager 路径的数据流

```
用户: y = relu(matmul(x, w))

1. matmul(x, w)
   ├── Ops 层: 调用 matmul_impl(x, w) → 计算结果 z
   └── Autograd 层: 创建 MatMulBackward, 保存 x 和 w, 挂到 z 上

2. relu(z)
   ├── Ops 层: 调用 relu_impl(z) → 计算结果 y
   └── Autograd 层: 创建 ReLUBackward, 保存 z, 挂到 y 上

此时 y.grad_fn → ReLUBackward → MatMulBackward → AccumulateGrad(x)
                                                → AccumulateGrad(w)
```

### 5.3 编译路径的数据流

```
用户: auto compiled = forge::compile(fn)

1. Graph Capture:
   ├── 创建 TracingContext
   ├── 创建代理张量 (ProxyTensor) 作为输入
   ├── 执行 fn(proxy_x, proxy_w)
   │   ├── matmul: 不计算，记录 Node{op=MatMul, inputs=[%0,%1]} → %2
   │   └── relu:   不计算，记录 Node{op=Relu, inputs=[%2]} → %3
   └── 标记 %3 为输出

2. Graph IR:
   graph(%0: f32[32,128], %1: f32[128,64]):
       %2 = matmul(%0, %1)  -> f32[32,64]
       %3 = relu(%2)        -> f32[32,64]
       return (%3)

3. Compiler: 运行优化 pass（此例无可优化项）

4. Runtime: compiled.run(real_x, real_w)
   ├── 绑定真实数据到 %0, %1
   ├── 执行 %2 = matmul(real_x, real_w)
   └── 执行 %3 = relu(%2)，返回结果
```

---

## 6. 关键设计决策总结

| 决策 | 我们的选择 | 理由 | 业界对照 |
|------|-----------|------|---------|
| C++ 标准 | C++20 | concepts + ranges + span，现代且实用 | PyTorch: C++17, XLA: C++17 |
| 构建系统 | Bazel + Bzlmod | 精确依赖管理，适合多层项目 | PyTorch: CMake, XLA: Bazel |
| 引用计数 | `shared_ptr` | 简洁，足够教学用途 | PyTorch: intrusive_ptr (更快) |
| 计算图 | 动态图 (define-by-run) | 调试友好，代码直觉 | PyTorch: 同; JAX: 函数变换 |
| 图捕获 | 纯 tracing | 最简单的实现方式 | PyTorch: Dynamo (更强大) |
| Graph IR | 自定义轻量 SSA | 避免 MLIR 依赖 | XLA: HLO; StableHLO: MLIR |
| 后端 | 仅 CPU | 聚焦架构理解，不分散精力 | 工业框架: CPU + GPU + TPU |

---

## 7. 项目结构

```
forge/
├── core/          # Layer 1: 张量表示与内存
├── ops/           # Layer 2: 算子实现
├── autograd/      # Layer 3: 自动微分引擎
├── graph/         # Layer 4+5: 图捕获与 IR
├── compiler/      # Layer 6: 优化 pass
├── runtime/       # Layer 7: 执行调度
├── nn/            # 上层: 神经网络模块 (Linear, ReLU 等)
├── optim/         # 上层: 优化器 (SGD, Adam)
├── common/        # 公共工具: 错误处理、日志、类型定义
examples/          # 端到端示例
tests/             # 单元测试与集成测试
benchmarks/        # 性能基准
docs/              # 设计文档 (你在这里)
```

---

## 8. 实现路线图

| 阶段 | 内容 | 里程碑 |
|------|------|--------|
| **Phase 1** | Tensor Core + 基本 Ops | `randn({2,3}) + ones({2,3})` 能跑通 |
| **Phase 2** | Autograd | `loss.backward()` 能计算梯度，gradcheck 通过 |
| **Phase 3** | Graph Capture + IR | `forge::compile(fn)` 能生成并 dump 计算图 |
| **Phase 4** | Compiler Passes | 常量折叠、DCE 等 pass 能优化图 |
| **Phase 5** | Runtime | 编译后的图能实际执行，结果与 eager 一致 |
| **Phase 6** | 端到端 Demo | MLP 训练 + Tiny Transformer 推理 |
