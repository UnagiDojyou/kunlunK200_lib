# Resident F16 Attention primitives (0.6.0)

These APIs operate on caller-owned device memory. They contain no GGML logic and do not change the common Q4 weight ABI or persistent weight allocations.

## Views and matrix multiplication

`Tensor4Layout` holds four logical extents, four byte strides and F32/F16 storage. Inner elements must be contiguous; outer dimensions may be permuted or padded. Extents are positive, at most65536, non-overlapping, element aligned and bounded by INT_MAX byte spans. Zero/negative strides and overlapping view rows are rejected.

`attention_matmul` computes:

```
Y[n,m,h,b] = sum_k A[k,n,h/(Hq/Hkv),b/(Bq/Bkv)] * B'[k,m,h,b]
```

The output is F32. Head and batch broadcasting use integer groups; Hq/Hkv and Bq/Bkv must be integral. K,N<=16384, M<=256, Hq<=128 and Bq<=32. All pointer spans, alignment, address overflow, output/input overlap and owning Kernel PD are checked before submission. Input pointers must belong to the supplied PD.

F16 values are expanded on device. With `round_b_to_f16=true` and F16 A, B is rounded to F16 before expansion, matching a half-input CPU reference. Other F32 inputs retain F32 precision. Products and eight-lane partial accumulators are F32; no INT8 quantization is introduced. Final lane reduction is F32. Different reduction order can produce small differences from CPU SIMD.

The CL16 kernel uses8x8 output tiles,128-element K tiles and reuses A across grouped query heads. Active input views are gathered on device where needed. Verified KL1 SDK casts convert to F32 scratch once per matrix call, rather than repeating conversion in each arithmetic tile. `KUNLUN_ATTENTION_PACK=cl` selects the slower software pack for reference comparisons; the default is the verified SDK cast path. The SDK canonicalizes F16 negative zero to F32 positive zero. Finite values, subnormals and FP32-to-F16 rounding were checked independently; NaN payload identity is not promised.

Each F32 pack buffer is limited to64 MiB. The reusable F16 staging buffers can add32 MiB each. Thus matrix pack storage can retain up to192 MiB per Kernel, plus small launch metadata. This is transient workspace, not a persistent FP32 KV shadow. Larger unsupported matrix shapes are rejected before launch.

## Softmax

`attention_softmax` applies F32 scale, an optional F32/F16 additive mask, optional positive ALiBi bias, stable row maximum subtraction, exponentiation and normalization. Head/batch mask dimensions repeat with modulo indexing. Mask query rows cover all input query rows. Negative-infinite mask entries produce zero probability; an entirely masked row returns zeros. The CPU implementation may treat an entirely masked row as undefined, so that case is compared against an explicit zero reference.

Input/output are F32; row width<=16384. Finite logits and parameters are required, except negative infinity used for masking. The stable CL implementation is retained. The tested SDK Softmax did not meet the selected precision/mask behavior and is not a production path. There is no Softmax HBM workspace or activation readback.

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
