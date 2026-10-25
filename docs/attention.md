# Resident F16 Attention primitives (0.6.3)

These APIs operate on caller-owned device memory. They contain no GGML logic and do not change the common Q4 weight ABI or persistent weight allocations.

## Views and matrix multiplication

`Tensor4Layout` holds four logical extents, four byte strides and F32/F16 storage. Inner elements must be contiguous; outer dimensions may be permuted or padded. Extents are positive, at most65536, non-overlapping, element aligned and bounded by INT_MAX byte spans. Zero/negative strides and overlapping view rows are rejected.

`attention_matmul` computes:

```
Y[n,m,h,b] = sum_k A[k,n,h/(Hq/Hkv),b/(Bq/Bkv)] * B'[k,m,h,b]
```

The output is F32. Head and batch broadcasting use integer groups; Hq/Hkv and Bq/Bkv must be integral. K,N<=16384, M<=256, Hq<=128 and Bq<=32. All pointer spans, alignment, address overflow, output/input overlap and owning Kernel PD are checked before submission. Input pointers must belong to the supplied PD.

The CL reference path expands F16 values on device. With `round_b_to_f16=true` and F16 A, B is rounded to F16 before expansion, matching a half-input CPU reference. Other F32 inputs retain F32 precision. Products and eight-lane partial accumulators are F32; no INT8 quantization is introduced. Final lane reduction is F32. Different reduction order can produce small differences from CPU SIMD.

The CL16 kernel uses8x8 output tiles,128-element K tiles and reuses A across grouped query heads. Active input views are gathered on device where needed. Verified KL1 SDK casts convert to F32 scratch once per matrix call, rather than repeating conversion in each arithmetic tile. `KUNLUN_ATTENTION_PACK=cl` selects the slower software pack for reference comparisons; the default is the verified SDK cast path. The SDK canonicalizes F16 negative zero to F32 positive zero. Finite values, subnormals and FP32-to-F16 rounding were checked independently; NaN payload identity is not promised.

Each F32 pack buffer is limited to64 MiB. The reusable F16 staging buffers can add32 MiB each. Thus matrix pack storage can retain up to192 MiB per Kernel, plus small launch metadata. This is transient workspace, not a persistent FP32 KV shadow. Larger unsupported matrix shapes are rejected before launch.


## SDNN matrix multiplication (0.6.1)

The default `KUNLUN_ATTENTION_MATMUL=auto` policy uses the direct SD program for half A, half-rounded B, K and N multiples of 32, dense F32 Y and 32-byte aligned base pointers. M may be 1 or have a partial final 32-row tile. For measured long-context GQA KQ GEMV (M=1, K=64, N>=1024, Hkv<Hq), auto/default uses CL while KQV stays on SDNN. This improves the measured TinyLlama long-context TG at the cost of retaining CL F32 scratch and another SDK context. `sdnn` forces the supported SD path for comparisons; `cl` or an unrecognized value uses the existing CL reference. Unsupported SD shapes continue through CL without changing the public layout contract. `GGML_KUNLUN_ATTENTION` remains the backend opt-in; this does not enable resident Attention when it is disabled.

This is the KL1 floating interface implemented with symmetric INT16 conversion, rather than a claim of IEEE half multiplication in the MAC array. DMA converts half HBM values to INT16, `mm_int16` / `mm_acc_int16` use dequantization scales and the visible F32 L1E accumulator, then EW/RS/DMAOUT write F32. Numerical tests distinguish the visible accumulation from half accumulation and half output. Internal multiplier/partial-adder precision is not documented here. INT16 conversion adds quantization error; select `cl` when its arithmetic is required.

`kunlun-attention-sdnn.xpu` assigns a 32-column KV tile to one of four SDNNs, keeps the complete K slice sequence in L1W (at most 1 MiB), and reuses it across query tiles and grouped heads/batches. K is processed in pieces of at most 256. Only valid query rows are written. The program uses the existing 8-context SD launch and one control context per SDNN.

The half KV allocation is read directly when its physical span is compact. Otherwise it is gathered into a half buffer. Half input is never expanded into a full F32 KV buffer on the SD path. Compact F32 Q is cast in physical order while its original strides are retained, avoiding an F32 gather. Other supported Q/probability views gather as required and cast once to half. The legacy `kunlun-attention-range.xpu` computes one half-rounded B maximum for each head/batch/32-query tile; 0.6.3 adds the DMA route described below. The SDK maximum for A has four partial results with `ncluster=4`; all four must be reduced. Its first entry is not a global maximum.

All workspaces and SDK contexts remain owned by the Kernel's PD and are reused and released by RAII. Statistics include the half buffers, range records and SDK-max buffer, but not all allocations internal to the SDK. Prior CL buffers may remain retained if a Kernel has also executed fallback shapes. Softmax, RoPE, cache writes and the common Q4 ABI are unchanged.

Set `KUNLUN_ATTENTION_TIMING=1` only for diagnostics. It prints completed core-launch and full-API durations without adding waits. Whole-model performance must omit this variable, operation tracing and profiling preload. The API still has its normal completion waits.

## Softmax

`attention_softmax` applies F32 scale, an optional F32/F16 additive mask, optional positive ALiBi bias, stable row maximum subtraction, exponentiation and normalization. Head/batch mask dimensions repeat with modulo indexing. Mask query rows cover all input query rows. Negative-infinite mask entries produce zero probability; an entirely masked row returns zeros. The CPU implementation may treat an entirely masked row as undefined, so that case is compared against an explicit zero reference.

Input/output independently support F32/F16; row width<=16384. F16 I/O is added in 0.6.2 (the prior API supported F32 I/O and F32/F16 masks). Finite logits and parameters are required, except negative infinity used for masking. NaN/positive infinity is outside the finite-input guarantee. These exceptional inputs have legacy CL behavior that differs from the host reference; the tested new path preserves those original CL results.

For width<=3072 the default kernel holds a whole F32 row in local memory, applies scale/mask and normalization with eight-lane SIMD, and evaluates native exp after max subtraction. A dense non-masked interval skips negative-infinite tails; arbitrary holes retain element checks. Eight independent exp instructions are unrolled while the sum order remains sequential. Only the final result is rounded for F16 output. The compiler checks the complete local stack against the 16256-byte limit. Outer strides and mask broadcasting are unchanged; exact in-place aliasing is supported.

`KUNLUN_ATTENTION_SOFTMAX=legacy` selects the original F32 three-pass kernel. Width>3072 automatically uses it. F16 I/O uses a separate streaming fallback that recomputes exp on its final pass, so no half intermediate is rounded. All paths have one CL launch and the normal API completion wait; cached metadata updates may have an additional upload wait. The original F32 kernel is unchanged. The tested SIMD polynomial exp was slower than native exp on PP and is excluded from the production build. SDK Softmax did not meet the selected precision/mask behavior and is not a production path. There is no activation readback or HBM intermediate workspace; each used program retains a 256-byte configuration allocation.

## RoPE and contiguous output

`attention_rope` supports NORMAL and NEOX, F32/F16, partial rotary width and an even offset. Width<=512, batch dimension3=1. YaRN extrapolation and external frequency factors remain the caller's fallback responsibility. The phase and rotation kernels use a reusable F32 trig buffer of `n_dims * tokens *4` bytes.

The function reads only the I32 position IDs back to validate range reduction; it does not read Q/K activations back. Within absolute `position*freq_scale`<=32768, trig calculation and rotation run on device. Outside that interval, the small trig table is computed on the host and uploaded; rotation still runs on device and `rope_host_trig` records the fallback. This guard avoids using an unvalidated polynomial range.

`attention_contiguous` copies a supported four-dimensional view to a dense allocation in logical element order, with matching F32/F16 type. It supports the permuted KQV output. Partial overlap is rejected. An already dense exact alias is a no-op.

## Lifetime, safety and diagnostics

Calls serialize with the existing runtime mutex and complete before returning. Cached programs, SDK Contexts, staging and workspaces are owned by Kernel RAII and one PD. Buffers are reused as shapes change and released on destruction. Attention reserves no explicit L3 buffer. SDK allocation observations must be reported separately from explicit library workspace.

The private160-byte launch configuration is checked with shared size/offset assertions and copied into a cached256-byte device buffer. KL1 does not accept the original by-value record or the oversized scalar launch payload. Metadata H2D transfers are distinct from activation/KV transfers. Completion waits are retained; this is not an asynchronous Attention API.

`*_reason` functions return an empty string for supported layout contracts. API calls return false before arithmetic submission for invalid contracts. Runtime/kernel errors retain the existing fail-stop behavior. `attention_statistics` reports matrix/Softmax/CONT/RoPE calls, pack calls, explicit workspace bytes and host trig fallbacks. CPU model quality must be checked separately: small floating differences can cross existing downstream A8 quantization boundaries.

## Build and tests

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target k200-attention-bench k200-attention-contract
# pd D S M KV-heads Q-heads storage-bytes past row-gap batches
build/k200-attention-bench 0 64 257 7 4 32 2 128 16 1
build/k200-attention-bench 1 128 2048 1 32 32 2 2040 0 1
```

Use the established thermal guard, persistent abort latch and device-state checks for every device run. The benchmark checks CPU references and guard bytes, then measures seven completed calls after warmup. The contract test uses rejected inputs and should be run with the supplied no-device shim when auditing the absence of device submissions.

The compiler checks each new CL assembly stack frame before producing its image. Keep source, shared configuration header, image and entry metadata together. SDKs, models and generated images/logs do not belong in Git.

Softmax-only comparison (seven completed API calls after warmup):

```
cmake --build build -j4 --target k200-softmax-bench
# pd N Q heads batches input-bytes output-bytes mask-bytes pattern gap permute inplace
build/k200-softmax-bench 0 512 128 32 1 4 4 2 1 0 0 0
build/k200-softmax-bench 1 257 7 6 2 2 4 4 13 16 1 0
```

Patterns cover ordinary values (0), causal mask (1), mask holes/all-masked first row (2), alternating all-masked rows (3), large values (4), exp underflow range (5), constant values (6), Inf/NaN (7-9), negative infinity (10), zero/negative scale (11-12), and ALiBi (13). Nonfinite cases use the original CL result as the compatibility oracle and report CPU special-value differences separately.

## SDNN preprocessing (0.6.3)

The default `KUNLUN_ATTENTION_PREP=auto` retains SDK FP32-to-FP16 casting and A/KV gathering and findmax. `kunlun-attention-range-dma.xpu` scans the original B view with one-dimensional SDNN DMA findmax, then rounds its absolute maximum to F16 once, preserving the original scale order. It writes the same aligned 32-byte record per head/batch/32-query tile. It uses no MAC operations and no activation readback. Contiguous query rows are combined into large scans, capped at 65536 elements per DMA request; every partial result contributes to the maximum. TG rows and contiguous B rows use this route. Interleaved PP query rows retain the faster CL range; outer strides not divisible by32 also retain the old path. `dma` forces the DMA range for aligned supported views, and `legacy` preserves the previous preprocessing. Unknown modes use legacy preprocessing. The matmul `auto/sdnn/cl` policy and long-context GQA KQ fallback are unchanged.

Casts on the SDK default stream are enqueued without an immediate wait. The existing gather and final preprocessing waits cover the independent work; MAC is submitted only after A maxima, B ranges and half inputs are ready. Cached metadata is still waited on after uploads, and the public API still waits for completed MAC output before return. Input/metadata/workspace cannot be reused by the caller until the API returns. Contexts are privately owned by Kernel/PD, and a non-default SDK stream forces the immediate cast wait. `KUNLUN_ATTENTION_PREP_PIPELINE=0` retains the immediate wait for comparisons. A maxima are recomputed on every call, reducing all four SDK partial maxima in the unchanged MAC program; no pointer-based KV cache of ranges is introduced.

This change moves range reduction off scalar CL. It does not claim a reduced logical HBM payload or fewer launches: SDK cast and range still read B independently, and the half buffer remains. A DMA cast/findmax fusion candidate did not pass API numerical validation and is excluded. A two-dimensional DMA findmax probe timed out and is not used. F16 magnitude scalar candidates were slower and are excluded.

`KUNLUN_ATTENTION_PREP_TRACE=1` is diagnostic-only: it adds a wait after SDK findmax and retains the cast wait to separate stage times. `KUNLUN_ATTENTION_PREP_VERIFY=1` is also diagnostic-only: it downloads original/packed data and scale records to check half rounding, all four A maxima and values exceeding the chosen scale. Never enable either flag, matmul timing, operation tracing, state diagnostics or profiling preloads in normal performance runs.

Build `k200-attention-prep-bench` for old/new direct comparisons. Arguments are `pd op D S M HK HQ B-bytes pattern gap batches layout` (`op=0` KQ, `op=1` KQV). Pattern0 is ordinary data,1 zeros,2 a last-head outlier,3 half subnormals,4 varying head scales,5 half-rounding ties,6 FP16 cast underflow,7 outliers in the first and last chunks. Layout0 is dense,1 head-interleaved Q,2 also pads V's inner storage. `KUNLUN_PREP_MUTATE=1` checks changed inputs and a changed query view on the same Kernel. These tests preserve input bytes and output guard/padding. Example:

```
build/k200-attention-prep-bench 0 1 128 512 128 32 32 4 0 0 1 2
```
