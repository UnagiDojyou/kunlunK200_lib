# Large-query PP (0.7.11, opt-in)

Set KUNLUN_LARGE_PP=1 in both the library and GGML process. Unset, 0, and other values retain the existing query/placement limits. The evaluated adapter configuration is GGML_KUNLUN_ATTENTION=1, GGML_KUNLUN_FLASH=1 and GGML_KUNLUN_ROPE=1.

Flash accepts up to 1024 queries under this option, subject to the existing D64/128, F16 KV, F32 output, broadcast, stride, mask, scale, integer-extent and SRAM/workspace checks. More than 128 queries are processed as completed calls over query views of at most 128. Q axis1, output axis2, and mask axis1 are offset together. KV views and mask head/batch dimensions are unchanged. Absolute positions are encoded in the supplied mask, not inferred from a chunk-local row. Each chunk retains the existing query pack, online Softmax, KQ/PV, join and stream completion rules. No device kernel or public ABI changes.

Workspace validation uses the live query chunk. Larger head/batch combinations that exceed the old 16 MiB workspace or input SRAM budget are still rejected. GGML still rejects sinks, soft cap, unsupported types/layouts and other existing restrictions. Disabling the option restores the original Flash M256 bound and Attention-chain CPU placement policy.

The adapter permits validated RoPE and CONT beyond the old blanket query-count policy while Flash is enabled. RoPE uses the original phase/rotation kernels, positions, parameter and alias checks. Non-unit element strides are explicitly rejected because the row kernel reads contiguous elements. CONT retains the original gather/copy and overlap validation. Position-ID readback for RoPE range validation remains; Q/K payload readback is not added.

Tracked KV writes from 257 through 1024 rows use the original writer in chunks of at most 256, including I64 indices. Chunks avoid a one-row tail so PP cannot enter the special TG pending-maximum schedule. The outer runtime mutex hides provisional metadata updates until the whole write completes. The host targets are a unique set, not an index-order mapping. Bounds, uniqueness, F16 rounding and nonfinite detection remain checked. Poisoned caches keep the original fallback, including after a later finite write. Unsupported cache layouts retain generic SET_ROWS and conservative invalidation.

Q4 weight ABI, Row A8, Activation Lease generations, GEMV/GEMM, SwiGLU and TG dispatch are unchanged. In particular, the large-query option does not relax Lease identity or lifetime checks.

KUNLUN_LARGE_PP_STATS=1 prints logical large-query batches and completed query chunks. GGML_KUNLUN_DIAGNOSTICS=1 prints CPU-placement reasons. Keep preload observers, eval tracing and verification-only settings out of normal timing.

The optional k200-large-pp-test target checks manual-query bitwise equivalence and sampled independent FP64 reference, masks/ALiBi/GQA/batch/stride, RoPE aliasing, CONT, and finite/nonfinite KV metadata across I32/I64 boundaries. It requires KUNLUN_BUILD_BENCHMARKS=ON. The Japanese evaluation in the parent repo records actual model performance and precision; CPU Attention is not bitwise equivalent to the existing F16/I16 Flash arithmetic.
