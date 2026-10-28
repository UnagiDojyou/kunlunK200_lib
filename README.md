# KunlunK200: KL1 device library and benchmarks

Fork repository: [kunlunK200_lib](https://github.com/UnagiDojyou/kunlunK200_lib).

A fork of [sam6321/KunlunK200](https://github.com/sam6321/KunlunK200), extended with an independent C++ library. It can be used without GGML, llama.cpp, or model files. The original guide is [README.upstream.md](README.upstream.md).

## Responsibilities

* `include/kunlun/k200.h`, `src/`: allocation, transfers, P2P, F32/F16/Q4_0 matrix multiplication, rowwise INT8 multiplication with two-pass activation correction, and elementwise operations.
* `kernels/`: device source for SIMD matrix and elementwise operations. Compiled device code is generated outside the source tree.
* `bench/library-bench.cpp`: library-only CPU-reference checks and timings for GEMV/GEMM, memory transfers, ADD, MUL, SILU, RMS normalization and scaling.
* `bench/k200_*`, `bench/kernels/`: original SDK/CDNN INT4/INT8/INT16 GEMM and pipeline experiments. These benchmark targets access the SDK directly; they are separate from the library API.
* `tools/thermal-guard.py`: bounded execution, die/HBM temperature monitoring, persistent stop latch at 80 C.

The kernel driver belongs to the separate `kunlun_k200` fork. The GGML adapter, KV-cache policy, model loading and layer/row splitting belong to the separate `llama.cpp` fork. This library has no dependency on either source tree.

## Dependencies and build

Tested on x86_64 Ubuntu Server 26.04.1, kernel 7.0.0-38-generic, K200 KL1, XRE 4.33.0, matching KL1 XDNN (20220919), and native x86 XTDK 2.7.98.2. Use the XDNN version identified in [THIRD_PARTY.md](THIRD_PARTY.md); newer XDNN packages can have incompatible APIs.

Keep SDK installations outside this repository. `XRE` contains `include/` and `lib64/`, `XDNN` contains `include/` and `so/`, and `XTDK_DIR` contains `bin/clang-10` and the XPU builtins. The validated native package is [XTDK 2.7.98.2](https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/release_paddle/2.7.98.2/xtdk-llvm15-ubuntu1604_x86_64.tar.gz).

```bash
export XRE=/path/to/xpu-4.33.0
export XDNN=/path/to/xdnn
export XTDK_DIR=/path/to/xtdk-llvm15-ubuntu1604_x86_64
cmake -S . -B ../build/library -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/kunlun"
cmake --build ../build/library -j4
cmake --install ../build/library
```

Alternatively, set `KUNLUN_PREBUILT_KERNELS` to an external directory containing the two validated `.bin`/`.entry` pairs. The loader resolves installed kernels relative to `libkunlunk200.so`; `KUNLUN_KERNEL_DIR` is an optional runtime override. Moving the install prefix does not require recompilation. XRE and XDNN remain external shared-library dependencies.

## CMake consumer

```cmake
find_package(KunlunK200 0.1 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE KunlunK200::kunlunk200)
```

Configure with `-DCMAKE_PREFIX_PATH=/path/to/install`. Public headers do not include vendor SDK headers. See [examples/library](examples/library) for a separate consumer project.

The API accepts device pointers and row-major arrays. Matrix multiplication computes `Y[batch,n] = X[batch,k] * transpose(W[n,k])`; a batch of one is GEMV. K must be a multiple of 32 and no greater than 16384. Each `Kernel` is bound to the first PD on which it is used. `matmul()` submits work; call `select(pd)` followed by `check(wait())` before reading results or freeing inputs. `elementwise()` completes before returning. Use `runtime_mutex()` around sequences of raw `select`, `copy` and `wait`, because device selection and launch arguments are process global. Allocation and high-level operations already take that recursive lock.

`allocate()` returns null on allocation failure. `Buffer` and checked operations print an error and abort on failure; this is currently a fail-fast API, not a recoverable status API. PD0 and PD1 are the only supported device indices. Individual allocations should stay at or below 1 GiB to avoid KL1 HBM address-region limits.

Q4_0 uses 18-byte blocks (F16 scale plus 16 low/high nibble pairs for 32 values). It is distinct from the vendor CDNN INT4 format. `rowwise_i8` adds weight quantization and stages activation quantization on the CPU; it changes results and uses more device memory than Q4_0. Compare accuracy as well as speed. `upload_i8_rows()` takes F32 host weights and builds the packed allocation used by this path.

## Benchmarks

```bash
export KUNLUN_LOG_DIR=/path/outside/repos/results
guard="$HOME/.local/kunlun/bin/k200-thermal-guard"
bench="$HOME/.local/kunlun/bin/k200-bench"
"$guard" --name memory --seconds 60 -- "$bench" memory
"$guard" --name gemv --seconds 60 -- "$bench" matmul q4_0 0 4096 4096 1
"$guard" --name gemm --seconds 60 -- "$bench" matmul i8 1 1024 512 4
"$guard" --name rms --seconds 60 -- "$bench" rms
```

`k200-bench OP FORMAT PD K N BATCH` checks every output against a CPU reference. Formats are `f32`, `f16`, `q4_0`, `i8`; operations also include `pipeline`, `memory`, `add`, `multiply`, `silu`, `rms`, `scale`. `pipeline` alternates matrix multiplication and in-place SILU with the same Kernel instance and checks the combined output. Default dimensions are 256, 65, 4, including an output-row tail. The memory test checks H2D, D2D, P2P and D2H. Original benchmarks are built as `k200-gemm-bench`, `k200-int4-bench`, `k200-pipe-bench`, `k200-smoke` and `k200-vvadd`. Their native kernel experiments retain their original documented command-line options; they are not part of the public ABI.

Never continue running after the temperature guard creates `THERMAL_ABORT`. Kernel errors also require stopping clients and checking both PD states. The driver fork documents the card-wide reset procedure. No reset is performed automatically by this library.

## Source and binary policy

New source is covered by the existing MIT license. Vendor components retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md). SDKs, model files, logs and builds are external. This fork removes the formerly tracked XDNN payload from its current tree; upstream Git history still contains those files. The upstream ancestry is intentionally preserved.

## Native Q4_0 (opt-in)

Version 0.3 adds a standalone native Q4_0 API without changing existing matmul defaults. See [native Q4 API, constraints, benchmarks and source provenance](docs/native-q4.md). Native preserves Q4 weights at approximately 4.5 bpw; CDNN remains faster in the measured B1 cases.

## Optional common Q4_0 matrices

Version 0.4 adds a typed, versioned common-F weight handle for GEMV and SD GEMM. Existing APIs and defaults are preserved. See [the ABI, ownership and dispatch guide](docs/common-q4.md) for build commands, limitations, diagnostics and the independent broadcast opt-in.

See [F16 indexed cache updates](docs/set-rows.md).

See [Fused SwiGLU](docs/swiglu.md).

Native resident F16 Attention is available as an explicit opt-in; see [Attention implementation, constraints and measurement notes](docs/attention.md).

Version 0.6.1 adds direct SDNN Attention matrix multiplication with half KV inputs and F32 visible accumulation/output. The CL reference remains selectable; see [precision, layout and dispatch details](docs/attention.md#sdnn-matrix-multiplication-061).

Version 0.6.2 adds the local-memory CL Softmax path for rows up to 3072 elements, independent F16/F32 I/O, and `KUNLUN_ATTENTION_SOFTMAX=legacy` for comparison/fallback. Native exp and the previous SDNN matrix path are retained; see [Softmax details](docs/attention.md#softmax).

Version 0.6.3 adds a dedicated SDNN DMA range pass and a shared preprocessing completion wait. MAC arithmetic and fast CL Softmax are retained. `KUNLUN_ATTENTION_PREP=legacy` selects the previous preprocessing. See [SDNN preprocessing](docs/attention.md#sdnn-preprocessing-063).

## Experimental FlashAttention (0.7.0)

The opt-in SDNN/CL block Attention path uses shared L3 for scores, probabilities and FP32 online state. See [configuration, contracts and comparison controls](docs/flash-attention.md). The separate Attention path remains the default.

### 0.7.4: exact KV maximum reuse

`KUNLUN_FLASH_KV_MAX=auto` (default) reuses GPU maxima only for backend-owned,
tracked finite F16 storage and proven-zero append slots. `legacy` retains SDK
findmax; `verify` independently compares cache hits with SDK results. Unknown
writes/restore, alias and unsupported layouts safely fall back. KQ/PV, Online
Softmax and quantization arithmetic are unchanged.

Auto starts at S512 for D*HK >= 1024, and S8192 for 256 <= D*HK < 1024.
Smaller features are opt-in only. Capacity below the threshold also retains the
original writer. See [Japanese evaluation](docs/reports/kv-max-0.7.4.ja.md) and
[reproduction](docs/reports/kv-max-0.7.4-reproduce.ja.md).

### 0.7.5: SDNN/CL pipeline

Version 0.7.5 overlaps next-block KQ with ordered online softmax using two Score/P slots and separate streams. Automatic selection is limited to the measured D128 and long-context D64 TG families; PP and other shapes remain sequential. See [pipeline controls, ownership and benchmarks](docs/flash-pipeline.md).

Additional bit-preserving KL1 Online Softmax is available explicitly with `KUNLUN_FLASH_ONLINE=simd8`; defaults are unchanged. See [docs/flash-online-extra.md](docs/flash-online-extra.md).

<!-- KV_DMA_076 -->

## FlashAttention K/V DMA 0.7.6

`KUNLUN_FLASH_KV_DMA=auto/legacy/coalesced` selects the validated TG path or the retained transfer kernels. Small-KV-head decoding experiments use explicit `decode2/decode4`. Online auto remains Ordered. See [evaluation](docs/flash-kv-dma.md) and [reproduction](docs/flash-kv-dma.md#build-and-comparison).

## Common Q4 Activation snapshots (0.7.7)

The optional prepared-activation API reuses identical quantization results across QKV and FFN gate/up projections. It keeps common weight ABI v1 and the validated 0.7.6 kernels unchanged. See [ownership, limits and benchmarks](docs/q4-activation-sharing.md). Existing matmul APIs remain available.
