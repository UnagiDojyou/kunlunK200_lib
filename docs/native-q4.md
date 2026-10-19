# Native Q4_0 (opt-in integration)

KunlunK200 0.3 adds a GGML-independent native Q4_0 API. Existing matmul, WeightType, Buffer, Kernel, Statistics, CMake targets and defaults are preserved.

## API and storage

Call native_q4_layout_supported(K,N) and native_q4_size(K,N) before allocation. pack_native_q4(dst,size,source,K,N) accepts contiguous row-major 18-byte Q4_0 blocks: little-endian FP16 scale, followed by 16 bytes holding values j and j+16. Source and destination must not overlap. upload_native_q4(device,dst,source,K,N) packs and synchronously uploads a full matrix.

Each N32 tile stores [K32 block, adjacent-K pair, N lane] signed INT4 bytes, then unchanged FP16 block scales. The mapping is reversible: the original mathematical values are preserved; the offset-8 Q4_0 nibble representation is converted to signed INT4 and adjacent K values are paired. No weight requantization is performed. N is padded to 32; payload and scale regions are aligned to 256 bytes. Complete N32 matrices use 4.5 bits per weight plus at most alignment padding.

NativeQ4Workspace binds to one PD on first use. native_q4_matmul(workspace,PD,Y,W,X,K,N,batch,InputType) returns true for accelerated SD execution and false for exact CL fallback. Both paths complete before return. Inputs and outputs stay in device memory. F32 output, contiguous F32/F16 input, and finite input values are required. Buffers must not overlap. The weight pointer must have 256-byte alignment; F32 and F16 pointers must have element alignment. SD additionally requires 64-byte input/output alignment.

The host layout policy accepts K divisible by 32, 32 <= K <= 16384, N > 0, K*N <= 90,000,000. This retains the validated experiment envelope; larger matrices are not silently accepted. SD accepts batch=1 and N divisible by 128: N divisible by 256 uses compact-n64-2d, the other N128 cases use the validated N32 pipeline. The native Q4 CL path supports other row counts and batches (subject to INT_MAX byte/index bounds) without CPU transfers or a second resident weight copy. It computes with the original Q4 values; its rounding differs from the W4A8x2 approximation. native_q4_reason reports why SD is unavailable; pointer alignment is checked separately at execution.

## Lifetime and synchronization

The existing process-wide recursive runtime_mutex protects selection, argument setup, allocation, all launches and completion. Each PD has an allocator-owned L3 partial buffer shared through reference-counted workspace ownership. A call completes before any other call reuses or grows it. Destruction and growth wait before freeing memory. No fixed L3 address is reserved. Only XRE NOMEM/NOSUPPORT allocation errors select fallback; other errors abort before any further device operation. If the XRE allocator cannot grant L3, a persistent HBM partial buffer is used and logged once. That workspace retains HBM mode until destruction, avoiding repeated failed allocations. Code and HBM scratch are lazy and reused; no steady-state weight rearrangement or CPU activation quantization occurs. The moving zero tail is cleared on XPU so one workspace can safely alternate K values.

Error handling follows the existing fail-fast library convention: runtime/kernel errors identify the API, device and diagnostic tensor and abort. The caller must use the thermal guard, retain stop latches, preserve the failed artifacts and logs, and perform the existing safe recovery before another device operation. This API does not reset the card or change the driver, firmware or clocks.

NativeQ4Workspace::statistics reports SD/CL call counts and logical buffer lengths. A 64-byte parameter block occupies a minimum 256-byte allocation, adding 192 bytes to HBM footprint. L3 storage is shared per PD: do not sum the same shared buffer across workspace statistics. allocated_bytes tracks ordinary library buffers, while code and SDK-internal allocations need API profiling or external memory inspection.

## Build and benchmark

Build with the existing XRE, XDNN and XTDK variables. All native kernels are compiled from source by the existing compile-native.sh and installed beside the shared library. Header dependencies are included in CMake. KUNLUN_PREBUILT_KERNELS must contain the new bin/entry pairs too; keep their source/compiler provenance.

~~~bash
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S repos/KunlunK200 -B build/native-integration-lib -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-integration-lib -j4 --target k200-native-q4-bench k200-native-q4-state-bench k200-bench
export KUNLUN_LOG_DIR=$K200_WORKSPACE/results/native-q4-local
mkdir -p "$KUNLUN_LOG_DIR"
k200-thermal-guard --name native-q4 --seconds 120 -- build/native-integration-lib/k200-native-q4-bench native 0 4096 4096 1 f32 3 7
# Modes: native, direct, cdnn, cpu. Arguments after mode: PD K N batch dtype pattern trials.
# cpu measures the decoded-weight FP64 reference using 6 OpenMP threads, not optimized GGML Q4 inference.
~~~

KUNLUN_Q4_WEIGHT_FILE optionally selects an exact row-major Q4_0 fixture. The benchmark checks every scale byte and every nibble through the inverse mapping, verifies all outputs against FP64 CPU reference, separates packing/upload/validation/cold-call time, and reports all warm trials plus median/min/max/standard deviation. The setup_ms field excludes the inverse-layout validation cost, which is separately reported as layout_check_ms. State tests alternate K and N32/N64 shapes, share scratch across contexts, and verify misaligned SD pointers use CL.

## Provenance and adopted optimizations

Baseline experiment: codex/k200-native-next-20261019, implementation/verification commit 36a9c8218fad36339410511c7a97b5cba8bb0260; final documentation afd99db. Production baseline f2e29b8fe13a4d74a8345da633a7f6c7f900e7f5.

| Experiment source | Production source | Treatment |
|---|---|---|
| next-bulk-n64-2d.xpu | kernels/kunlun-native-q4.xpu | Same N64 schedule and guards; local include path |
| next-bulk-n64-pipeline.h | kernels/native-q4/pipeline.h | Same integer MAC, scale slots, DMA/DS/RS dependencies |
| next-bulk.xpu / next-bulk-pipeline.h | kunlun-native-q4-n32.xpu / native-q4/pipeline-n32.h | Existing safe N32 path for N128 splits |
| activation-stage.h | kernels/native-q4/activation-stage.h | Bounded DS fences, 2D DMA/minimal clear for N64 |
| quant-max-64 / quant-main-compact-64 | kunlun-native-q4-max / main | chunk64, CL16, same rounding |
| quant-residual-compact-64 | kunlun-native-q4-residual | Same residual math; add device zero-tail clear for varying K |
| int4-pipe/q4-sum-partial.xpu | kunlun-native-q4-sum.xpu | Same FP32 accumulation, CL16 |
| resident-bench panel rearrangement | src/q4-layout.cpp | Direct, reversible conversion from public Q4_0 blocks |
| resident-bench scratch/loader | src/k200.cpp | Integrate existing loader, RAII buffers, PD ownership and L3 synchronization |

No experimental host executable or GGML code is linked into the library. The compact CL fallback is a small production decoder for the same resident layout. The SDK helper subset comes from the already tracked bench/kernels/cdnn_ref.h and retains its provenance; no SDK binaries are added.

Not adopted: retired vsum and unbounded staged kernels (714/timeouts), pass/hoist/fold variants (slower or unresolved), SD reduction (unverified), and newer compact-n64-dual16 research (different bank layout, dual-engine routing and L2 footprint; deferred until production resource/dispatch validation). No experimental branch is overwritten. Q4_K is not implemented.

## Validation snapshot (2026-10-19)

Llama 2 7B Q4_0 blk.0 attn_q/ffn_up/ffn_down fixtures; B1/F32, 900 MHz, OMP_WAIT_POLICY=PASSIVE. Each timing combines 3 processes x 7 warm trials. CPU gold is decoded Q4 weights with FP64 accumulation. Full min/max/stddev/IQR and trials are retained in the integration evidence, including slower native results.

| K x N | PD | Native median ms | CDNN median ms | Native NMSE |
|---|---:|---:|---:|---:|
| 4096 x 4096 | 0 | 0.338455 | 0.232140 | 8.704e-11 |
| 4096 x 11008 | 0 | 0.712449 | 0.431599 | 2.442e-10 |
| 11008 x 4096 | 0 | 0.843438 | 0.636038 | 2.340e-10 |
| 4096 x 4096 | 1 | 0.330048 | 0.229371 | 8.704e-11 |
| 4096 x 11008 | 1 | 0.739542 | 0.428676 | 2.442e-10 |
| 11008 x 4096 | 1 | 0.840245 | 0.613637 | 2.340e-10 |

4096 x 4096 native retains 9,437,184 B of weights, 37,952 B of logical HBM scratch (38,144 B after the 256-byte allocation minimum; input/output excluded), and 2,097,152 B of L3 partials. API HBM peak is 9,520,896 B, compared with CDNN 83,979,008 B including SDK internal allocations. Native has 5 launches/3 waits and no steady H2D/D2H/D2D/P2P copies per GEMV. These are per-matrix values, not whole-model memory ratios.

Both PDs passed F32/F16, zero/outlier/wide-exponent, K-tail, N32/N64, B8/B32 CL fallback, irregular row partitions, workspace sharing/K changes, and element-aligned F32/F16 input and F32 output fallback tests. A test-only allocator shim returns XPUERR_NOMEM for L3; the implementation retains HBM partials, logs once, and makes only one failed L3 attempt across warmup and seven trials. No hardware reset or induced device fault is involved.

The mixed-resource regression alternates N32/N64 native and existing CDNN on both PDs in both first-call orders, including L3 growth from 2,818,048 to 5,636,096 bytes while the SDK context stays alive. All outputs passed CPU checks.
