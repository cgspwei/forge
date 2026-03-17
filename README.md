# Forge

**A miniature tensor runtime and graph compiler in modern C++, built with Bazel.**

Forge is a small but rigorous machine learning systems project that implements a complete pipeline from eager tensor execution to graph-based compilation and optimized CPU execution.

------

## ✨ Highlights

- Modern C++ tensor core with strided memory layout
- Reverse-mode autodiff (dynamic computation graph)
- Graph capture from eager execution
- Compact graph IR with explicit dataflow
- Compiler passes:
  - constant folding
  - dead code elimination
  - algebraic simplification
  - (optional) operator fusion
- CPU execution backend with parallel kernels
- Basic memory planning and buffer reuse
- End-to-end demos:
  - MLP training
  - Tiny transformer block

------

## 🧠 Motivation

Modern ML frameworks (e.g., PyTorch 2.x, XLA-based systems) are converging toward a common architecture:

```
eager frontend → graph capture → IR → compiler passes → backend execution
```

Forge is a minimal, self-contained reimplementation of this idea.

The goal is not feature completeness, but to:

- understand deep learning runtime internals from first principles,
- explore graph-based compilation and optimization,
- and practice modern C++ and systems design.

------

## 🏗 Architecture Overview

Forge is structured as a layered system:

```
Tensor → Autograd → Graph → Compiler → Runtime → Backend
```

### 1. Tensor Core

- strided tensor representation
- storage / view separation
- shape, dtype, and device abstraction

### 2. Autograd Engine

- dynamic computation graph
- reverse-mode differentiation
- gradient accumulation

### 3. Graph Capture

- trace eager execution into graph
- explicit node / edge representation
- metadata propagation (shape, dtype)

### 4. Graph IR

- op-based intermediate representation
- SSA-like dataflow
- typed edges

### 5. Compiler Passes

- constant folding
- dead node elimination (DCE)
- algebraic simplification
- (optional) pattern-based fusion

### 6. Runtime & Execution

- CPU backend
- thread pool for parallel execution
- kernel dispatch system
- (optional) memory planning & buffer reuse

------

## 🚀 Examples

### Eager execution + autograd

```cpp
Tensor x = randn({32, 128}, requires_grad=true);
Tensor w = randn({128, 64}, requires_grad=true);

Tensor y = relu(matmul(x, w));
Tensor loss = mean(y);

loss.backward();
```

------

### Graph capture and compilation

```cpp
auto fn = [](Tensor x, Tensor w) {
    return relu(matmul(x, w));
};

auto compiled = forge::compile(fn);

Tensor out = compiled.run(x, w);
```

------

## 📦 Build (Bazel)

Forge uses **Bazel with Bzlmod**.

### Build example

```bash
bazel build //examples:mlp_train
```

### Run example

```bash
bazel run //examples:mlp_train
```

### Run tests

```bash
bazel test //tests/...
```

### Run benchmarks

```bash
bazel run //benchmarks:matmul_bench
```

------

## 📁 Project Structure

```
forge/
  core/        # tensor, storage, shape
  ops/         # primitive operations
  autograd/    # differentiation engine
  graph/       # graph IR and builder
  compiler/    # optimization passes
  runtime/     # executor, thread pool
  nn/          # neural network layers
  optim/       # optimizers
examples/      # demos
benchmarks/    # performance tests
tests/         # unit tests
docs/          # design docs
```

------

## 📊 What This Project Demonstrates

Forge is designed to showcase:

- How tensor computation is represented and executed
- How reverse-mode autodiff works internally
- How eager programs are lowered into a graph IR
- How compiler passes optimize computation graphs
- How a CPU runtime schedules and executes kernels
- How memory reuse can be implemented in a graph-based runtime

------

## ⚠️ Non-Goals

Forge is intentionally **not**:

- a production ML framework
- a GPU / CUDA system
- a distributed training system
- a full LLM serving engine
- a drop-in replacement for PyTorch or TensorFlow

------

## 🔭 Future Work

- KV cache and inference-time optimization
- graph-level scheduling strategies
- GPU backend
- distributed execution
- Python bindings

------

## 📚 References

- PyTorch 2.x compiler stack
- XLA / StableHLO
- Autograd systems
- ML runtime and compiler design

------

## 👤 Author

This project is built as a systems-focused ML infrastructure exploration, with emphasis on:

- deep learning runtimes
- graph compilers
- modern C++ design
- performance-oriented systems engineering
