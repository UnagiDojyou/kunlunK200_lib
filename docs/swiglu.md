# Fused device SwiGLU (0.5.0)

`swiglu(Kernel&, device, output, output_bytes, gate, gate_bytes, up, up_bytes, layout)` computes `SiLU(gate)*up`. The device pointers belong to one owning PD. `SwiGLULayout` describes width, flattened row count, three byte row strides and `InputType::f32` or `f16`. Inputs and output use the same type. F16 arithmetic has F32 intermediates and one final round-to-nearest-even conversion.

Inner elements must be contiguous and element aligned. Width is 1..65536; positive row counts and all required byte spans must fit INT_MAX. Strides must cover a row. The API checks byte extents, address overflow and output/input overlap. Read-only gate/up may share a packed allocation. Exact output alias is allowed with the same row stride, including provably disjoint gate/up segments in packed rows. Other partial alias is rejected.

`swiglu_reason(layout)` returns an empty string for a supported shape/type/stride. `swiglu` returns false before device submission for unsupported layouts or pointer contracts. Runtime/kernel errors use the existing fail-stop `check()` policy. The call holds the runtime mutex and completes before returning; a caller can immediately enqueue the down projection on the same PD. Kernel code is cached in `Kernel` and released by its existing RAII destructor.

The CL16 kernel uses three 256-element float/half unions (3072 bytes). FP16 expansion runs backward and output packing runs forward so the buffers can be reused safely. There is no HBM/L3 workspace, CPU activation transfer or intermediate SiLU HBM tensor. Contiguous rows use a flat tiled path; row-padded and packed views use the same row-stride path. Each tile reads gate/up once and writes output once.

Nonfinite input classification is explicit because K200 scalar arithmetic does not preserve all IEEE NaN cases. The implementation matches the `x/(1+exp(-x))*up` classification, including negative-infinite gate and zero-times-infinity NaNs. NaN payload preservation is not promised. Finite F32 math is close to scalar libm, but not bitwise identical to every CPU SIMD approximation. In an A8 model, small differences can cross quantization thresholds and affect later logits. Validate model quality independently.

## Build and tests

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target k200-swiglu-bench k200-swiglu-contract
# pd width rows element-bytes pattern packed row-gap-bytes output-alias
build/k200-swiglu-bench 0 11008 128 4 0 0 0 0
build/k200-swiglu-bench 1 257 33 2 0 1 32 0
```

Run device commands under the existing thermal guard. The benchmark checks CPU-reference error, guard bytes and row padding, then times 15 completed calls after warmup. Its pass field is an explicit numerical assessment; a process return of zero alone is not a numerical pass. Alias tests check one update and omit destructive repeated timing.

`cmake/compile-native.sh` checks the SwiGLU assembly stack frame before generating the image. The tested XTDK reports initial local SP=16256; the final kernel frame is3136. An early 18496-byte frame caused714 and is rejected by this build guard. This check applies only to the new CL kernel and does not change existing SD images.

## 0.7.12 reevaluation

The CPU default is retained. With current FlashAttention and other common-Q4 optimizations, XPU SwiGLU improves the measured PP/TG workloads, but scalar/SIMD rounding and A8 threshold sensitivity still prevent bitwise model equivalence. See the 2026-10-30 defaults report; this does not change the supported types/layouts or 0/1/pp semantics.
