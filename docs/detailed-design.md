# Forge 详细设计文档

> **版本**: v0.1-draft
> **范围**: Layer 1–4 (Tensor Core → Autograd → Graph Capture → Graph IR)
> **C++ 标准**: C++20
> **构建系统**: Bazel with Bzlmod

---

## 目录

1. [设计原则](#1-设计原则)
2. [Layer 1: Tensor Core](#2-layer-1-tensor-core)
3. [Layer 2: Autograd Engine](#3-layer-2-autograd-engine)
4. [Layer 3: Graph Capture](#4-layer-3-graph-capture)
5. [Layer 4: Graph IR](#5-layer-4-graph-ir)
6. [跨层设计](#6-跨层设计)
7. [业界实现对比](#7-业界实现对比)
8. [附录](#8-附录)

---

## 1. 设计原则

### 1.1 核心理念

| 原则 | 说明 |
|------|------|
| **值语义优先** | Tensor 对象采用 shared_ptr + COW (Copy-on-Write)，对外表现为值语义 |
| **分层解耦** | 每层通过明确定义的接口交互，下层不依赖上层 |
| **零开销抽象** | 使用 C++20 concepts 约束模板，编译期消除虚函数开销 |
| **显式优于隐式** | 设备、dtype、内存布局均显式声明，不做隐式转换 |
| **可测试性** | 每个组件可独立单元测试，通过接口注入依赖 |

### 1.2 C++20 特性使用策略

| 特性 | 用途 |
|------|------|
| `concepts` | 约束 Op 接口、Backend 接口、数值类型 |
| `std::span` | 无拷贝的 shape/stride 传递 |
| `std::format` | 统一的日志和错误消息格式化 |
| `ranges` | 形状计算、元素遍历 |
| `designated initializers` | 配置结构体的可读初始化 |
| `三路比较 (<=>)` | Shape、DType 的比较 |
| `constexpr` 扩展 | 编译期 shape 推导和 dtype 大小计算 |

---

## 2. Layer 1: Tensor Core

### 2.1 概述

Tensor Core 是整个系统的基础数据层，负责多维数组的表示、存储管理和基本操作。

### 2.2 类型系统

#### 2.2.1 DType（数据类型）

```cpp
enum class DType : uint8_t {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    // 预留扩展
};

// 编译期类型映射
template <DType D> struct dtype_traits;
template <> struct dtype_traits<DType::Float32> {
    using type = float;
    static constexpr size_t size = 4;
    static constexpr const char* name = "float32";
};
// ... 其他特化

// Concept: 可用于 Tensor 计算的类型
template <typename T>
concept TensorScalar = std::is_arithmetic_v<T> && !std::is_same_v<T, char>;
```

**设计决策**: 使用 `enum class` 而非 `std::variant` 作为运行时类型标识。理由：
- 枚举值可作为 switch 分派，编译器可检查完整性
- 内存占用仅 1 字节
- 编译期 traits 映射提供类型安全

#### 2.2.2 Device（设备抽象）

```cpp
enum class DeviceType : uint8_t {
    CPU,
    // 未来扩展: CUDA, Metal
};

struct Device {
    DeviceType type;
    int index;  // 设备编号，CPU 始终为 0

    constexpr auto operator<=>(const Device&) const = default;

    static constexpr Device cpu() { return {DeviceType::CPU, 0}; }
};
```

### 2.3 形状与步幅

#### 2.3.1 Shape

```cpp
class Shape {
public:
    // 最大维度数 (避免动态分配的 small-buffer 优化)
    static constexpr size_t kMaxDims = 8;

    // 构造
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::span<const int64_t> dims);

    // 属性
    [[nodiscard]] size_t ndim() const noexcept;
    [[nodiscard]] int64_t operator[](size_t i) const;
    [[nodiscard]] int64_t numel() const noexcept;   // 元素总数
    [[nodiscard]] bool is_scalar() const noexcept;   // ndim == 0

    // 视图
    [[nodiscard]] std::span<const int64_t> dims() const noexcept;

    // 比较
    constexpr auto operator<=>(const Shape&) const = default;

private:
    std::array<int64_t, kMaxDims> dims_{};
    uint8_t ndim_ = 0;
};
```

**设计决策**: 使用固定大小数组 + `kMaxDims = 8`。
- 主流深度学习张量很少超过 8 维（常见为 2-5 维）
- 避免 `std::vector` 的堆分配，Shape 操作是极高频路径
- 与 PyTorch 的 `SmallVector<int64_t, 5>` 类似策略

#### 2.3.2 Strides（步幅）

```cpp
class Strides {
public:
    static constexpr size_t kMaxDims = Shape::kMaxDims;

    Strides() = default;
    explicit Strides(std::span<const int64_t> strides);

    // 从 shape 计算默认行优先步幅
    static Strides contiguous(const Shape& shape);

    [[nodiscard]] int64_t operator[](size_t i) const;
    [[nodiscard]] std::span<const int64_t> values() const noexcept;
    [[nodiscard]] bool is_contiguous(const Shape& shape) const noexcept;

private:
    std::array<int64_t, kMaxDims> strides_{};
    uint8_t ndim_ = 0;
};
```

### 2.4 存储层

#### 2.4.1 Storage（内存缓冲区）

```cpp
class Storage {
public:
    // 不可默认构造（必须指定大小）
    Storage(size_t size_bytes, Device device = Device::cpu());

    // 不可拷贝，可移动
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;

    // 访问
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] size_t size_bytes() const noexcept;
    [[nodiscard]] Device device() const noexcept;

    // 类型化访问
    template <TensorScalar T>
    [[nodiscard]] std::span<T> as() noexcept;

    template <TensorScalar T>
    [[nodiscard]] std::span<const T> as() const noexcept;

private:
    std::unique_ptr<uint8_t[]> data_;
    size_t size_bytes_;
    Device device_;
};
```

**设计决策**: Storage 不可拷贝，只能通过 `shared_ptr<Storage>` 共享。
- 与 PyTorch 的 `StorageImpl` 设计一致
- 避免意外的大块内存拷贝
- 多个 Tensor view 可共享同一 Storage

#### 2.4.2 Storage 与 View 的关系

```
Tensor A (shape=[2,3], strides=[3,1], offset=0)
    │
    ├──► shared_ptr<Storage>  ◄──┤
    │    [data: 24 bytes]        │
    │                            │
Tensor B = A.transpose(0,1)     │
    (shape=[3,2], strides=[1,3], offset=0)
```

### 2.5 TensorImpl（内部实现）

```cpp
class TensorImpl {
public:
    TensorImpl(std::shared_ptr<Storage> storage,
               Shape shape, Strides strides,
               int64_t storage_offset, DType dtype);

    // 元数据访问
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Strides& strides() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] int64_t storage_offset() const noexcept;
    [[nodiscard]] int64_t numel() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;

    // 数据访问
    [[nodiscard]] void* data_ptr() noexcept;
    [[nodiscard]] const void* data_ptr() const noexcept;

    template <TensorScalar T>
    [[nodiscard]] T* data() noexcept;

    // Autograd 接口 (Layer 2 使用)
    [[nodiscard]] bool requires_grad() const noexcept;
    void set_requires_grad(bool requires_grad);
    [[nodiscard]] std::shared_ptr<class AutogradMeta> autograd_meta() const;
    void set_autograd_meta(std::shared_ptr<class AutogradMeta> meta);

private:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Strides strides_;
    int64_t storage_offset_ = 0;
    DType dtype_;

    // Autograd 元数据 (惰性分配)
    bool requires_grad_ = false;
    std::shared_ptr<AutogradMeta> autograd_meta_;
};
```

### 2.6 Tensor（用户接口）

```cpp
class Tensor {
public:
    // --- 构造 ---
    Tensor() = default;  // 空张量
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    // --- 工厂函数 (自由函数) ---
    // 见 2.7 节

    // --- 元数据 ---
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] int64_t size(int dim) const;
    [[nodiscard]] size_t ndim() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] Device device() const;
    [[nodiscard]] int64_t numel() const;
    [[nodiscard]] bool is_contiguous() const;

    // --- 视图操作 (不拷贝数据) ---
    [[nodiscard]] Tensor view(Shape new_shape) const;
    [[nodiscard]] Tensor reshape(Shape new_shape) const; // 可能拷贝
    [[nodiscard]] Tensor transpose(int dim0, int dim1) const;
    [[nodiscard]] Tensor slice(int dim, int64_t start, int64_t end) const;
    [[nodiscard]] Tensor contiguous() const; // 若已连续则返回自身

    // --- 数据访问 ---
    template <TensorScalar T>
    [[nodiscard]] T item() const;  // 仅标量张量

    template <TensorScalar T>
    [[nodiscard]] const T* data_ptr() const;

    // --- Autograd ---
    [[nodiscard]] bool requires_grad() const;
    Tensor& set_requires_grad(bool requires_grad);
    [[nodiscard]] Tensor grad() const;
    void backward() const;  // 仅标量张量

    // --- 实现访问 ---
    [[nodiscard]] TensorImpl& impl() const;
    [[nodiscard]] std::shared_ptr<TensorImpl> impl_ptr() const;

    // --- 有效性检查 ---
    [[nodiscard]] bool defined() const noexcept;
    explicit operator bool() const noexcept;

private:
    std::shared_ptr<TensorImpl> impl_;
};
```

### 2.7 工厂函数

```cpp
namespace forge {

// 未初始化张量
Tensor empty(Shape shape, DType dtype = DType::Float32,
             Device device = Device::cpu());

// 零张量
Tensor zeros(Shape shape, DType dtype = DType::Float32);

// 全一张量
Tensor ones(Shape shape, DType dtype = DType::Float32);

// 正态分布随机张量
Tensor randn(Shape shape, DType dtype = DType::Float32);

// 均匀分布随机张量
Tensor rand(Shape shape, DType dtype = DType::Float32);

// 从数据创建 (拷贝)
template <TensorScalar T>
Tensor from_data(std::span<const T> data, Shape shape);

// 单位矩阵
Tensor eye(int64_t n, DType dtype = DType::Float32);

// 等差序列
Tensor arange(double start, double end, double step = 1.0,
              DType dtype = DType::Float32);

}  // namespace forge
```

### 2.8 算子系统

#### 2.8.1 Op 接口

```cpp
// Concept: 可分派的算子
template <typename Op>
concept TensorOp = requires(Op op) {
    { op.name() } -> std::convertible_to<std::string_view>;
};

// 算子注册表 (运行时分派)
enum class OpKind : uint16_t {
    // 一元运算
    Neg, Abs, Exp, Log, Sqrt, Relu, Sigmoid, Tanh,
    // 二元运算
    Add, Sub, Mul, Div, Pow, MatMul,
    // 归约运算
    Sum, Mean, Max, Min,
    // 形状运算
    Reshape, Transpose, Slice, Cat, Unsqueeze, Squeeze,
};
```

#### 2.8.2 逐元素运算

```cpp
namespace forge {

// 一元
Tensor neg(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);

// 二元 (支持广播)
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// 运算符重载
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a);  // 一元取负

// 归约
Tensor sum(const Tensor& input, std::optional<int> dim = std::nullopt,
           bool keepdim = false);
Tensor mean(const Tensor& input, std::optional<int> dim = std::nullopt,
            bool keepdim = false);

// 矩阵运算
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace forge
```

#### 2.8.3 广播规则

Forge 采用与 NumPy/PyTorch 相同的广播语义：

1. 从右向左对齐维度
2. 维度为 1 的可以扩展
3. 维度相同或其中一个为 1 才兼容

```cpp
// 广播形状推导
std::optional<Shape> broadcast_shapes(const Shape& a, const Shape& b);
```

### 2.9 内存布局与索引

```cpp
// 给定 shape=[d0, d1, ..., dn-1], strides=[s0, s1, ..., sn-1]
// 元素 (i0, i1, ..., in-1) 的字节偏移:
//   offset = storage_offset + sum(ik * sk) for k in [0, n)

// 行优先 (C-order) 默认步幅:
//   strides[n-1] = 1
//   strides[k]   = strides[k+1] * shape[k+1]   for k in [0, n-2]
```

### 2.10 关键流程

#### 创建张量

```
randn({2, 3})
  │
  ├── Shape shape({2, 3})      // ndim=2, numel=6
  ├── Strides strides = Strides::contiguous(shape)  // [3, 1]
  ├── Storage storage(6 * sizeof(float))   // 24 bytes
  ├── 填充随机数据到 storage
  └── Tensor(make_shared<TensorImpl>(storage, shape, strides, 0, Float32))
```

#### 视图操作 (transpose)

```
A = randn({2, 3})          // shape=[2,3], strides=[3,1]
B = A.transpose(0, 1)      // shape=[3,2], strides=[1,3], 共享 Storage
                            // 无数据拷贝!
```

---

## 3. Layer 2: Autograd Engine

### 3.1 概述

Autograd 引擎实现**动态计算图上的反向模式自动微分**。前向执行时记录操作到 DAG，`backward()` 时按拓扑逆序计算梯度。

### 3.2 核心数据结构

#### 3.2.1 AutogradMeta

```cpp
// 挂载在 TensorImpl 上的梯度元数据
struct AutogradMeta {
    Tensor grad;                           // 累积的梯度
    std::shared_ptr<GradFn> grad_fn;       // 产生此张量的反向函数
    uint32_t output_index = 0;             // 该张量是 grad_fn 的第几个输出
    bool requires_grad = false;

    // 梯度累加 (原子操作，支持未来多线程)
    void accumulate_grad(const Tensor& grad);
};
```

#### 3.2.2 Edge（计算图中的边）

```cpp
struct Edge {
    std::shared_ptr<GradFn> function;   // 目标节点
    uint32_t input_nr;                   // 该边连接到 function 的第几个输入

    bool is_valid() const noexcept { return function != nullptr; }
};
```

#### 3.2.3 GradFn（计算图中的节点）

```cpp
class GradFn : public std::enable_shared_from_this<GradFn> {
public:
    virtual ~GradFn() = default;

    // 反向传播: 给定输出梯度, 计算输入梯度
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) = 0;

    // 节点名称 (用于调试和可视化)
    [[nodiscard]] virtual std::string_view name() const = 0;

    // 该节点的输入边 (连接到上游节点)
    [[nodiscard]] const std::vector<Edge>& next_edges() const noexcept;
    void set_next_edges(std::vector<Edge> edges);

    // 节点序号 (拓扑排序用)
    [[nodiscard]] uint64_t sequence_nr() const noexcept;

private:
    std::vector<Edge> next_edges_;
    uint64_t sequence_nr_;  // 全局递增序号

    static inline std::atomic<uint64_t> next_sequence_nr_{0};
};
```

#### 3.2.4 AccumulateGrad（叶子节点）

```cpp
class AccumulateGrad : public GradFn {
public:
    explicit AccumulateGrad(std::weak_ptr<TensorImpl> variable);

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
    std::string_view name() const override { return "AccumulateGrad"; }

private:
    std::weak_ptr<TensorImpl> variable_;
};
```

### 3.3 反向函数注册机制

每个前向算子需要注册对应的反向函数：

```cpp
// 示例: AddBackward
class AddBackward : public GradFn {
public:
    // 前向时保存必要的上下文
    // add(a, b) 的梯度: grad_a = grad_out, grad_b = grad_out
    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        auto& grad_out = grad_outputs[0];
        return {grad_out, grad_out};  // 梯度直接传递
    }

    std::string_view name() const override { return "AddBackward"; }
};

// 示例: MatMulBackward
class MatMulBackward : public GradFn {
public:
    // matmul(A, B) 的梯度:
    //   grad_A = grad_out @ B^T
    //   grad_B = A^T @ grad_out
    Tensor saved_a;  // 保存前向输入用于反向计算
    Tensor saved_b;

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
    std::string_view name() const override { return "MatMulBackward"; }
};

// 示例: ReLUBackward
class ReLUBackward : public GradFn {
public:
    // relu(x) 的梯度: grad_x = grad_out * (x > 0)
    Tensor saved_input;

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override;
    std::string_view name() const override { return "ReLUBackward"; }
};
```

### 3.4 前向执行与计算图构建

```cpp
// 以 add 为例的完整前向逻辑:
Tensor add(const Tensor& a, const Tensor& b) {
    // 1. 计算前向结果
    Tensor result = add_impl(a, b);  // 纯计算，无 autograd

    // 2. 若需要梯度，构建计算图
    if (a.requires_grad() || b.requires_grad()) {
        auto grad_fn = std::make_shared<AddBackward>();

        // 构建边: 连接到 a 和 b 的梯度源
        std::vector<Edge> edges;
        edges.push_back(edge_of(a));  // 若 a 是叶子则指向 AccumulateGrad
        edges.push_back(edge_of(b));
        grad_fn->set_next_edges(std::move(edges));

        // 将 grad_fn 关联到输出张量
        set_grad_fn(result, grad_fn, /*output_index=*/0);
    }

    return result;
}
```

### 3.5 反向执行引擎

```cpp
class Engine {
public:
    // 从 root 节点开始反向传播
    void execute(const Tensor& root, const Tensor& grad_output = {});

private:
    // 拓扑排序 + 反向遍历
    void compute_dependencies(const std::shared_ptr<GradFn>& root);
    void backward_pass();

    // 节点 -> 待消费的输出梯度
    std::unordered_map<GradFn*, std::vector<Tensor>> grad_buffer_;

    // 拓扑排序结果 (按 sequence_nr 降序)
    std::vector<std::shared_ptr<GradFn>> sorted_nodes_;
};
```

#### 反向传播算法

```
1. 从 loss.grad_fn 开始，BFS 遍历图，计算每个节点的入度
2. 将 root 的梯度 (默认为标量 1.0) 放入 grad_buffer_[root_fn]
3. 按 sequence_nr 降序遍历:
   a. 从 grad_buffer_ 取出该节点的所有输入梯度并求和
   b. 调用 node->apply(accumulated_grads)，得到输出梯度列表
   c. 将输出梯度分发到对应的 next_edges
4. AccumulateGrad 节点将最终梯度累加到叶子张量的 .grad 上
```

### 3.6 关键流程图

```
前向执行:

  x (leaf, requires_grad=true)
  │
  ├── matmul(x, w) ──► z    [记录 MatMulBackward, 保存 x, w]
  │                    │
  │                    ├── relu(z) ──► a    [记录 ReLUBackward, 保存 z]
  │                    │              │
  │                    │              ├── mean(a) ──► loss  [记录 MeanBackward]
  │                    │              │
  w (leaf)             │              │

反向执行 (loss.backward()):

  loss.grad = 1.0
  │
  MeanBackward::apply({1.0})
  │ → grad_a = 1.0 / numel
  │
  ReLUBackward::apply({grad_a})
  │ → grad_z = grad_a * (z > 0)
  │
  MatMulBackward::apply({grad_z})
  │ → grad_x = grad_z @ w^T
  │ → grad_w = x^T @ grad_z
  │
  AccumulateGrad(x): x.grad += grad_x
  AccumulateGrad(w): w.grad += grad_w
```

### 3.7 梯度检查工具

```cpp
namespace forge::autograd {

// 数值梯度检查 (用于验证反向实现的正确性)
bool gradcheck(
    std::function<Tensor(std::vector<Tensor>)> fn,
    std::vector<Tensor> inputs,
    double eps = 1e-6,
    double atol = 1e-5,
    double rtol = 1e-3
);

}  // namespace forge::autograd
```

---

## 4. Layer 3: Graph Capture

### 4.1 概述

Graph Capture 层将动态 eager 执行**追踪 (trace)** 为静态计算图。用户在 `forge::compile()` 中传入一个函数，系统使用代理张量执行该函数，记录所有操作到图结构中。

### 4.2 追踪策略

Forge 采用 **Tracing-based capture**（基于追踪的捕获）：

1. 创建代理张量（ProxyTensor），携带符号形状但无真实数据
2. 执行用户函数，每个算子调用被拦截并记录到 GraphBuilder
3. 结果是一个完整的 Graph IR

```cpp
// 用户接口
auto fn = [](Tensor x, Tensor w) {
    return relu(matmul(x, w));
};

auto compiled = forge::compile(fn);  // 触发追踪
Tensor out = compiled.run(x_real, w_real);  // 使用真实数据执行
```

### 4.3 ProxyTensor（代理张量）

```cpp
class ProxyTensor {
public:
    ProxyTensor(NodeId node_id, Shape shape, DType dtype);

    // 元数据 (用于形状推导)
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;

    // 图中对应的节点 ID
    [[nodiscard]] NodeId node_id() const noexcept;

private:
    NodeId node_id_;
    Shape shape_;
    DType dtype_;
};
```

### 4.4 追踪模式与调度

```cpp
// 全局追踪状态 (thread_local，支持嵌套追踪)
class TracingContext {
public:
    static TracingContext* current();
    static bool is_tracing();

    // RAII 管理追踪范围
    class Guard {
    public:
        explicit Guard(TracingContext& ctx);
        ~Guard();
    };

    // 记录算子调用
    NodeId record_op(OpKind op,
                     std::span<const NodeId> inputs,
                     const Shape& output_shape,
                     DType output_dtype);

    // 记录输入占位符
    NodeId record_input(const Shape& shape, DType dtype);

    // 标记输出
    void mark_output(NodeId node_id);

    // 获取构建的图
    [[nodiscard]] std::unique_ptr<Graph> finalize();

private:
    std::unique_ptr<GraphBuilder> builder_;

    static thread_local TracingContext* current_;
};
```

### 4.5 算子追踪拦截

每个算子在执行时检查追踪模式：

```cpp
Tensor matmul(const Tensor& a, const Tensor& b) {
    // 如果在追踪模式中
    if (auto* ctx = TracingContext::current()) {
        // 形状推导
        auto out_shape = matmul_shape(a.shape(), b.shape());
        auto out_dtype = a.dtype();

        // 记录到图中
        NodeId result = ctx->record_op(
            OpKind::MatMul,
            {proxy_node(a), proxy_node(b)},
            out_shape,
            out_dtype
        );

        // 返回新的代理张量
        return make_proxy_tensor(result, out_shape, out_dtype);
    }

    // 非追踪模式: 正常 eager 执行
    return matmul_impl(a, b);
}
```

### 4.6 CompiledFunction

```cpp
class CompiledFunction {
public:
    // 使用真实数据执行编译后的图
    Tensor run(std::span<const Tensor> inputs) const;
    Tensor run(const Tensor& input) const;

    // 多参数便捷接口
    template <typename... Args>
    Tensor operator()(Args&&... args) const;

    // 访问底层图 (调试用)
    [[nodiscard]] const Graph& graph() const;

private:
    std::unique_ptr<Graph> graph_;
    // 未来: 编译后的执行计划
};
```

### 4.7 compile() 流程

```
forge::compile(fn)
  │
  ├── 1. 创建 TracingContext
  ├── 2. 为每个参数创建 ProxyTensor (形状从示例输入推导)
  ├── 3. TracingContext::Guard guard(ctx)
  ├── 4. fn(proxy_inputs...)    // 执行用户函数，所有 Op 被拦截
  ├── 5. ctx.mark_output(result_node)
  ├── 6. auto graph = ctx.finalize()  // 得到 Graph IR
  ├── 7. 运行 Compiler Passes (Layer 5, 后续实现)
  └── 8. 返回 CompiledFunction(graph)
```

### 4.8 追踪的局限性

| 局限 | 说明 | 缓解措施 |
|------|------|----------|
| 不支持数据依赖控制流 | `if tensor.item() > 0` 无法追踪 | 提供 `forge::cond()` 和 `forge::while_loop()` |
| 动态形状 | 追踪固定形状，不同形状需重追踪 | shape 作为图的输入参数 (v2 目标) |
| 副作用 | print、文件 IO 不被记录 | 文档说明，仅追踪纯计算 |
| Python 回调 | N/A (纯 C++) | - |

---

## 5. Layer 4: Graph IR

### 5.1 概述

Graph IR 是 Forge 计算图的**中间表示**，采用 **SSA (Static Single Assignment)** 风格的有向无环图。每个节点代表一个操作，边代表数据依赖。

### 5.2 核心数据结构

#### 5.2.1 NodeId

```cpp
// 强类型节点标识符
struct NodeId {
    uint32_t value;

    constexpr auto operator<=>(const NodeId&) const = default;

    // 哈希支持
    struct Hash {
        size_t operator()(NodeId id) const noexcept {
            return std::hash<uint32_t>{}(id.value);
        }
    };
};

static constexpr NodeId kInvalidNode{UINT32_MAX};
```

#### 5.2.2 Node

```cpp
struct Node {
    NodeId id;
    OpKind op;                          // 操作类型
    std::vector<NodeId> inputs;         // 输入节点
    Shape output_shape;                 // 输出形状
    DType output_dtype;                 // 输出数据类型

    // 可选: 常量值 (用于 constant folding)
    std::optional<Tensor> constant_value;

    // 可选: 操作属性
    NodeAttributes attrs;

    // 辅助方法
    [[nodiscard]] bool is_placeholder() const noexcept;
    [[nodiscard]] bool is_output() const noexcept;
    [[nodiscard]] bool is_constant() const noexcept;
};
```

#### 5.2.3 NodeAttributes（操作属性）

```cpp
// 不同操作可能携带的额外参数
struct NodeAttributes {
    // Reduce 操作
    std::optional<int> reduce_dim;
    bool keepdim = false;

    // Reshape 操作
    std::optional<Shape> target_shape;

    // Transpose 操作
    int dim0 = 0, dim1 = 1;

    // Slice 操作
    int64_t slice_start = 0, slice_end = -1;
    int slice_dim = 0;

    // 常量标量值
    std::optional<double> scalar_value;
};
```

#### 5.2.4 Graph

```cpp
class Graph {
public:
    Graph() = default;

    // --- 节点管理 ---
    NodeId add_node(OpKind op, std::vector<NodeId> inputs,
                    Shape output_shape, DType output_dtype,
                    NodeAttributes attrs = {});

    NodeId add_placeholder(Shape shape, DType dtype);
    NodeId add_constant(Tensor value);
    void mark_output(NodeId node);

    // --- 节点访问 ---
    [[nodiscard]] const Node& node(NodeId id) const;
    [[nodiscard]] Node& node(NodeId id);
    [[nodiscard]] size_t num_nodes() const noexcept;

    // --- 图结构查询 ---
    [[nodiscard]] std::span<const NodeId> inputs() const noexcept;
    [[nodiscard]] std::span<const NodeId> outputs() const noexcept;
    [[nodiscard]] std::vector<NodeId> users(NodeId id) const;  // 使用该节点的下游节点
    [[nodiscard]] bool has_users(NodeId id) const;

    // --- 拓扑遍历 ---
    [[nodiscard]] std::vector<NodeId> topological_order() const;

    // --- 变换 ---
    void replace_all_uses(NodeId old_node, NodeId new_node);
    void erase_node(NodeId id);  // 仅当无 user 时

    // --- 验证 ---
    [[nodiscard]] bool verify() const;  // 检查图的一致性

    // --- 序列化 & 调试 ---
    [[nodiscard]] std::string dump() const;     // 可读文本表示
    [[nodiscard]] std::string to_dot() const;   // Graphviz DOT 格式

private:
    std::vector<Node> nodes_;                    // 按 NodeId 索引
    std::vector<NodeId> input_nodes_;            // 图的输入
    std::vector<NodeId> output_nodes_;           // 图的输出

    // 反向边索引 (node -> 使用它的节点列表)
    std::unordered_map<NodeId, std::vector<NodeId>, NodeId::Hash> use_map_;
};
```

### 5.3 图的 SSA 性质

- **每个 NodeId 唯一定义一个值** — 节点不可重赋值
- **边隐式表示**: 由 `Node::inputs` 指向其他 NodeId
- **不可变语义**: 创建后的节点不修改（替换而非修改）

```
// 示例: relu(matmul(x, w)) 的 Graph IR

%0 = placeholder(shape=[32, 128], dtype=f32)     // x
%1 = placeholder(shape=[128, 64], dtype=f32)      // w
%2 = matmul(%0, %1)  : shape=[32, 64], dtype=f32
%3 = relu(%2)         : shape=[32, 64], dtype=f32
output(%3)
```

### 5.4 GraphBuilder

```cpp
class GraphBuilder {
public:
    GraphBuilder() = default;

    // 构建接口 (TracingContext 使用)
    NodeId add_input(Shape shape, DType dtype);
    NodeId add_op(OpKind op, std::vector<NodeId> inputs,
                  Shape output_shape, DType output_dtype,
                  NodeAttributes attrs = {});
    NodeId add_constant(Tensor value);
    void mark_output(NodeId node);

    // 完成构建
    [[nodiscard]] std::unique_ptr<Graph> build();

private:
    std::unique_ptr<Graph> graph_ = std::make_unique<Graph>();
};
```

### 5.5 Graph 验证规则

`Graph::verify()` 检查：

1. **类型一致性**: 每个节点的输入 dtype 符合算子要求
2. **形状一致性**: 输出形状与算子的形状推导规则匹配
3. **无悬空引用**: 所有 `Node::inputs` 中的 NodeId 都存在
4. **无环**: 图是 DAG
5. **输入/输出完整**: 至少有一个 placeholder 和一个 output
6. **SSA 性质**: 每个 NodeId 只被定义一次

### 5.6 Graph Dump 格式

```
graph(%x: f32[32, 128], %w: f32[128, 64]):
    %2 = matmul(%x, %w)         -> f32[32, 64]
    %3 = relu(%2)               -> f32[32, 64]
    return (%3)
```

### 5.7 与 Compiler Passes 的接口

Graph IR 设计为 Compiler Passes（Layer 5）的输入/输出格式：

```cpp
// Pass 接口 (Layer 5 详细设计后补充)
class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string_view name() const = 0;
    virtual bool run(Graph& graph) = 0;  // 返回是否修改了图
};
```

---

## 6. 跨层设计

### 6.1 内存模型

```
┌─────────────────────────────────────────┐
│                 用户空间                   │
│  Tensor (handle, 8 bytes, 值语义)         │
│    │                                     │
│    └──► shared_ptr<TensorImpl>           │
│           │                              │
│           ├── Shape, Strides, DType      │
│           ├── storage_offset             │
│           ├── shared_ptr<Storage>         │
│           │     └── unique_ptr<uint8_t[]>│ ← 实际数据
│           └── shared_ptr<AutogradMeta>   │
│                 ├── grad (Tensor)        │
│                 └── shared_ptr<GradFn>   │ ← 计算图节点
└─────────────────────────────────────────┘
```

**引用计数层次**:
- `Tensor` → `TensorImpl`: shared_ptr（多个 Tensor 可指向同一 impl，如别名）
- `TensorImpl` → `Storage`: shared_ptr（视图共享存储）
- `TensorImpl` → `AutogradMeta`: shared_ptr（惰性分配）
- `GradFn` → `GradFn`: shared_ptr（计算图结构）
- `AccumulateGrad` → `TensorImpl`: weak_ptr（避免循环引用）

### 6.2 线程安全模型

| 组件 | 线程安全级别 | 说明 |
|------|-------------|------|
| Tensor (handle) | 不可共享 | 按值传递，各线程持有独立 handle |
| TensorImpl | 只读共享安全 | 元数据创建后不可变，数据通过 Storage 访问 |
| Storage | 不可共享 | 通过 shared_ptr 管理生命周期，数据访问需外部同步 |
| AutogradMeta::grad | 原子累加 | `accumulate_grad()` 内部加锁 |
| GradFn::sequence_nr | 原子分配 | 全局 atomic counter |
| TracingContext | thread_local | 每个线程独立的追踪上下文 |
| Graph | 构建期单线程 | 构建完成后只读，可安全共享 |

### 6.3 错误处理策略

```cpp
namespace forge {

// 错误类型层次
class ForgeError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class ShapeError : public ForgeError {
    using ForgeError::ForgeError;
};

class DTypeError : public ForgeError {
    using ForgeError::ForgeError;
};

class IndexError : public ForgeError {
    using ForgeError::ForgeError;
};

class AutogradError : public ForgeError {
    using ForgeError::ForgeError;
};

class GraphError : public ForgeError {
    using ForgeError::ForgeError;
};

}  // namespace forge

// 断言宏 (release 模式也检查)
#define FORGE_CHECK(cond, ...)                                      \
    do {                                                            \
        if (!(cond)) {                                              \
            throw ::forge::ForgeError(std::format(__VA_ARGS__));    \
        }                                                           \
    } while (false)

#define FORGE_CHECK_SHAPE(cond, ...)                                \
    do {                                                            \
        if (!(cond)) {                                              \
            throw ::forge::ShapeError(std::format(__VA_ARGS__));    \
        }                                                           \
    } while (false)
```

### 6.4 命名空间结构

```
forge::               // 顶层: Tensor, 工厂函数, 算子函数
forge::autograd::     // Autograd 引擎: GradFn, Engine
forge::graph::        // Graph IR: Graph, Node, GraphBuilder
forge::detail::       // 内部实现, 不暴露给用户
```

### 6.5 文件结构映射

```
forge/
├── core/
│   ├── dtype.h              // DType, dtype_traits
│   ├── device.h             // Device, DeviceType
│   ├── shape.h / shape.cc   // Shape
│   ├── strides.h / strides.cc // Strides
│   ├── storage.h / storage.cc // Storage
│   ├── tensor_impl.h / tensor_impl.cc // TensorImpl
│   ├── tensor.h / tensor.cc // Tensor (用户接口)
│   └── scalar.h             // TensorScalar concept
├── ops/
│   ├── unary_ops.h / .cc    // neg, abs, exp, log, relu, sigmoid, tanh
│   ├── binary_ops.h / .cc   // add, sub, mul, div, matmul
│   ├── reduce_ops.h / .cc   // sum, mean, max, min
│   ├── shape_ops.h / .cc    // reshape, transpose, slice, cat
│   ├── factory_ops.h / .cc  // zeros, ones, randn, rand, arange
│   └── broadcast.h / .cc    // 广播形状推导
├── autograd/
│   ├── autograd_meta.h      // AutogradMeta
│   ├── edge.h               // Edge
│   ├── grad_fn.h / grad_fn.cc // GradFn 基类
│   ├── accumulate_grad.h / .cc
│   ├── backward_fns/        // 各算子的反向函数
│   │   ├── add_backward.h / .cc
│   │   ├── matmul_backward.h / .cc
│   │   ├── relu_backward.h / .cc
│   │   └── ...
│   ├── engine.h / engine.cc // 反向执行引擎
│   └── gradcheck.h / .cc    // 数值梯度检查
├── graph/
│   ├── node.h               // Node, NodeId, NodeAttributes
│   ├── graph.h / graph.cc   // Graph
│   ├── graph_builder.h / .cc // GraphBuilder
│   ├── tracing.h / tracing.cc // TracingContext, ProxyTensor
│   └── compiled_fn.h / .cc  // CompiledFunction
└── common/
    ├── error.h              // ForgeError 层次, FORGE_CHECK 宏
    ├── logging.h            // 日志工具
    └── types.h              // 公共类型定义
```

### 6.6 Bazel BUILD 结构

```python
# forge/core/BUILD
cc_library(
    name = "core",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = ["//forge/common"],
    visibility = ["//visibility:public"],
)

# forge/autograd/BUILD
cc_library(
    name = "autograd",
    srcs = glob(["*.cc", "backward_fns/*.cc"]),
    hdrs = glob(["*.h", "backward_fns/*.h"]),
    deps = ["//forge/core", "//forge/ops"],
    visibility = ["//visibility:public"],
)

# forge/graph/BUILD
cc_library(
    name = "graph",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = ["//forge/core", "//forge/ops"],
    visibility = ["//visibility:public"],
)
```

---

## 7. 业界实现对比

### 7.1 Tensor 层对比

| 维度 | **Forge** | **PyTorch** | **XLA/StableHLO** |
|------|-----------|-------------|---------------------|
| 核心类 | `Tensor` → `TensorImpl` → `Storage` | `Tensor` → `TensorImpl` → `StorageImpl` | `XlaLiteral` / `Shape` |
| 存储分离 | ✅ Storage/View 分离 | ✅ 相同设计 | ❌ 值语义，无 view |
| 步幅 | 固定数组 `array<int64_t, 8>` | `SmallVector<int64_t, 5>` | N/A (仅支持 dense layout) |
| 类型系统 | `enum class DType` | `ScalarType` enum + `TypeMeta` | `PrimitiveType` enum |
| 设备抽象 | `Device{type, index}` | `Device{type, index}` + DeviceGuard | implicit (编译到特定后端) |
| 引用计数 | `shared_ptr` | `intrusive_ptr` (更高效) | N/A (SSA 值语义) |
| 元素数限制 | 理论上 `int64_t` 范围 | 同 | 同 |

**Forge 的简化**:
- 使用 `shared_ptr` 替代 PyTorch 的 `intrusive_ptr`（牺牲少量性能换取简洁性）
- 固定 8 维上限，避免动态分配
- 不支持 stride layout 以外的内存格式（PyTorch 支持 channels_last 等）

### 7.2 Autograd 对比

| 维度 | **Forge** | **PyTorch** | **JAX** |
|------|-----------|-------------|---------|
| 图类型 | 动态 (define-by-run) | 动态 (define-by-run) | 函数变换 (jit → trace) |
| 反向节点 | `GradFn` 虚函数 | `Node` (原名 `Function`) 虚函数 | Jaxpr 中的 `primitive.vjp` |
| 边表示 | `Edge{shared_ptr<GradFn>, input_nr}` | `Edge{shared_ptr<Node>, input_nr}` | Jaxpr 中的变量引用 |
| 叶子累加 | `AccumulateGrad` | `AccumulateGrad` | 函数式返回梯度 |
| 拓扑排序 | `sequence_nr` 降序 | `sequence_nr` + `topological_nr` | N/A (Jaxpr 是静态的) |
| 高阶导数 | 不支持 (v1) | 支持 (grad_fn 的 grad_fn) | 原生支持 (函数变换组合) |
| saved tensors | 直接成员变量 | `SavedVariable` (带 version 检查) | 不需要 (纯函数) |
| 多线程引擎 | 单线程 (v1) | 多线程 GraphTask | N/A |

**Forge 的简化**:
- 不支持高阶导数（无需记录反向函数本身的计算图）
- 不实现 `SavedVariable` 的 version 检查（PyTorch 用此检测 in-place 修改）
- 单线程引擎，避免复杂的 `GraphTask` / `NodeTask` / `ReadyQueue` 架构
- 不支持 autograd hooks（`register_hook`）

### 7.3 Graph Capture 对比

| 维度 | **Forge** | **PyTorch (torch.compile)** | **XLA (LazyTensor)** | **JAX (jit)** |
|------|-----------|---------------------------|---------------------|---------------|
| 捕获方式 | 代理张量追踪 | TorchDynamo (字节码改写) + FX tracing | LazyTensor IR 累积 + 标记同步 | Jaxpr tracing (抽象值) |
| 控制流 | 不追踪 (纯追踪) | guard + graph break | 不追踪 (触发同步) | `jax.lax.cond` / `scan` |
| 动态形状 | 固定形状 | SymInt 符号整数 | 部分支持 | 编译期确定 |
| 复杂度 | 低 (~500 LoC) | 极高 (~50K+ LoC) | 高 (~10K LoC) | 中 (~5K LoC) |
| Side effect | 不处理 | 检测并 graph break | 不处理 | 禁止 |

**Forge 的简化**:
- 纯 tracing，无字节码分析
- 不支持 graph break（整个函数必须可追踪）
- 不支持 SymInt（未来 v2 目标）
- 追踪在 C++ 层完成，无需 Python 交互

### 7.4 Graph IR 对比

| 维度 | **Forge** | **PyTorch FX** | **XLA HLO** | **StableHLO** |
|------|-----------|----------------|-------------|---------------|
| 表示 | `Graph` + `Node` (SSA) | `Graph` + `Node` (SSA) | `HloModule` + `HloComputation` + `HloInstruction` | MLIR Dialect |
| 节点粒度 | 与 eager ops 1:1 | 与 Python ops 1:1 | 低级 (接近硬件) | 中级 (标准化 HLO) |
| 类型 | `OpKind` enum | `Node.op` (字符串 + 目标) | `HloOpcode` enum | MLIR Op 定义 |
| 属性 | `NodeAttributes` struct | `Node.kwargs` (Python dict) | `HloInstruction` 子类 | MLIR Attributes |
| 子图 | 不支持 (v1) | `GraphModule` 嵌套 | `HloComputation` 嵌套 | MLIR Region |
| 序列化 | 文本 dump + DOT | Python pickle + 文本 | Protobuf (HloProto) | MLIR 文本/bytecode |
| 验证 | `verify()` 方法 | `lint()` | `HloVerifier` | MLIR verifier |
| 变换接口 | `Pass::run(Graph&)` | `torch.fx.Interpreter` 子类 | `HloPassInterface` | MLIR Pass framework |

**Forge IR 的设计取舍**:
- **粒度**: 与 eager ops 对齐（类似 FX），而非 HLO 的低级表示
  - 优势: 追踪简单，1:1 映射
  - 劣势: 优化空间受限（如无法表示 fused kernel 的边界）
- **无子图**: 简化实现，但限制了 cond/while 的表示能力
- **无 MLIR**: 避免引入庞大的 MLIR 依赖，自定义轻量 IR

### 7.5 总结: Forge 的定位

```
              复杂度
                ▲
                │
    XLA/HLO ────┤  ● 工业级编译器
                │
    PyTorch  ────┤  ● 工业级框架
    2.x          │
                │
                │
    JAX     ────┤  ● 生产级函数变换
                │
                │
    Forge   ────┤  ● 教学级实现    ← 我们在这里
                │
                └────────────────────► 功能覆盖
```

Forge 刻意选择在每一层做最小可行实现，目标是：
1. **完整走通 pipeline**: 从 eager → graph → compile → execute
2. **代码可读性 > 性能**: 优先教学目的
3. **每层 < 2000 LoC**: 保持项目规模可控
4. **C++20 最佳实践**: 展示现代 C++ 在系统编程中的应用

---

## 8. 附录

### 8.1 依赖关系图

```
forge/common   ← 无外部依赖
    ▲
    │
forge/core     ← 依赖 common
    ▲
    │
forge/ops      ← 依赖 core
    ▲
    ├──────────────────┐
    │                  │
forge/autograd         forge/graph
    ← 依赖 core, ops   ← 依赖 core, ops
```

### 8.2 构建依赖 (第三方)

| 依赖 | 用途 | 引入方式 |
|------|------|----------|
| GoogleTest | 单元测试 | Bzlmod |
| Google Benchmark | 性能基准测试 | Bzlmod |
| fmtlib (可选) | 若编译器不完整支持 `std::format` | Bzlmod |

### 8.3 实现优先级

| 阶段 | 目标 | 预估 LoC |
|------|------|---------|
| Phase 1 | Tensor Core (Shape, Storage, TensorImpl, Tensor, 工厂函数) | ~1200 |
| Phase 2 | 算子 (一元、二元、归约、matmul、广播) | ~1000 |
| Phase 3 | Autograd (GradFn, Engine, 基本反向函数, gradcheck) | ~1500 |
| Phase 4 | Graph IR (Node, Graph, GraphBuilder, verify, dump) | ~800 |
| Phase 5 | Graph Capture (TracingContext, ProxyTensor, CompiledFunction) | ~600 |
| Phase 6 | 集成测试 + MLP demo | ~500 |
| **总计** | | **~5600** |

### 8.4 术语表

| 术语 | 定义 |
|------|------|
| **Eager execution** | 立即执行，每个操作调用时即计算结果 |
| **Graph capture** | 将 eager 执行记录为静态计算图 |
| **SSA** | Static Single Assignment，每个变量只赋值一次 |
| **Stride** | 在某一维度上移动一个元素所需跨越的存储元素数 |
| **View** | 共享同一底层 Storage 的不同形状/步幅的张量 |
| **GradFn** | 梯度函数，反向传播时被调用来计算输入梯度 |
| **Proxy tensor** | 不含真实数据的占位张量，用于追踪时记录操作 |
| **Pass** | 对 Graph IR 进行转换的编译器变换 |
| **DCE** | Dead Code Elimination，移除不影响输出的节点 |
