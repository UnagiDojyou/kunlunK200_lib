# K200 experimental FlashAttention (0.7.0)

This opt-in path executes `GGML_OP_FLASH_ATTN_EXT`. It does not change the default separate KQ/Softmax/KQV path.

```bash
export GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1
export KUNLUN_ATTENTION_MATMUL=auto KUNLUN_ATTENTION_SOFTMAX=fast
export KUNLUN_ATTENTION_PREP=auto KUNLUN_FLASH_KV=1
# Use -fa on with the normal llama.cpp tools.
# The kunlun-llm-smoke / kunlun-quality-eval tools use K200_FLASH=1 instead.
```

Disable FlashAttention with `-fa off` (or `K200_FLASH=0` for the test tools) and unset `GGML_KUNLUN_FLASH`. The old `KUNLUN_ATTENTION_MATMUL=auto/sdnn/cl` settings remain available on that separate path. The fused path always uses SDNN INT16 MACs.

## Data path

Q is rounded to FP16. Compact physical F16 K/V views are used directly; other outer strides are gathered in F16. SDK findmax scans active K and V once per API; every SDNN reduces all four SDK partial maxima. Native DMA range produces Q tile scales. All data and scales are recomputed for each API, including after KV updates or state restore.

For each KV block:

1. SDNN KQ writes FP32 scores to shared L3.
2. CL updates the FP32 running maximum and denominator, adds the preceding PV result, rescales the output numerator, and replaces the score tile with FP32 exponentials.
3. SDNN input DMA converts the exponentials directly to INT16 with the bound 1; PV writes an FP32 partial output to L3.

After the final block, CL adds the last partial output and divides by the denominator. An all-masked row returns zero. The CL kernel uses the existing native exp, not a new approximate exp implementation. Score and probability share the same allocation. A private HBM scratch allocation provides a functional fallback when L3 cannot be allocated.

The wide PV kernel loads the complete feature width into L1W, reuses each probability tile across all output features, and bounds each input DMA request to 128 KiB of L2. TG GQA maps shared query heads to MAC rows; it preserves each head's individual Q scale during input conversion and restores that scale in CL.

`xpu_launch_async` calls use the default stream. Their order supplies inter-stage dependencies. Only the final completion wait is required on a warm API with unchanged metadata. Buffer growth and metadata replacement wait before reuse. The public API is synchronous and holds `runtime_mutex` throughout. L3 is shared with the existing Q4 scratch pool and is not reserved once per layer.

## API and limits

`flash_attention(Kernel &, ...)` takes `Flash4Layout`:

- Q `[D,M,Hq,B]`, F32 or F16; K/V `[D,S,Hkv,Bkv]`, F16.
- Y `[D,Hq,M,B]`, F32. The feature dimension must be contiguous; non-overlapping outer strides are accepted.
- Additive mask `[S,>=M,Hmask,Bmask]`, optional F16/F32; head/batch mask broadcast and ALiBi are supported.
- KV heads and batches repeat in contiguous groups, as in `ggml_mul_mat`.
- D 64/128, M <= 256, S a multiple of 32 and <= 16384; Hq <= 128, B <= 16. The selected SRAM and workspace bounds must also fit.
- Q/K/V bases must be 32-byte aligned. Inputs must remain finite after FP16 conversion; mask entries may also be negative infinity. NaN and positive infinity input semantics are not defined by this experimental SDNN API.
- Sinks, logit soft cap, other inner layouts/types and shapes outside these bounds are declined by the Kunlun backend. GGML can place the fused operation on CPU. Disable FA to select the original separate XPU path explicitly.

The maximum supported S is an API bound, not a claim of LLM validation at 16384 tokens. Hardware tests cover S up to 4096, and LLM generation covers prompt 2048 plus 64 generated tokens in a context of 4096.

For r=M*Hq*B and KV block size b, the L3 workspace is `r*(4*b+8*D+32)` bytes. D128/M128/H32/b512 uses 12.125 MiB. Q cast/gather scratch and SDK metadata are separate. Noncompact KV views may require additional F16 gather scratch. `flash_statistics` reports library scratch, not the complete model allocation or SDK-internal memory.

The Q scale covers the selected query tile; changing its size changes INT16 quantization. Bitwise agreement with the original 32-query tiles is not promised. Use query tile 32 to retain that scale granularity. Visible MAC outputs, online state and output accumulation are FP32; this does not specify the undocumented physical accumulator inside the MAC array.

## Comparison controls

| Setting | Values / default |
| --- | --- |
| `KUNLUN_FLASH_BLOCK` | 128/256/512/1024; default PP 512, TG 1024 (512 for >64 shared heads), clipped to S |
| `KUNLUN_FLASH_QUERY_TILE` | 16/32/64/128; default 128 |
| `KUNLUN_FLASH_PV` | `narrow` for the 32-feature comparison kernel; default wide |
| `KUNLUN_FLASH_TG_GROUP` | 0 disables TG head grouping; default enabled |
| `KUNLUN_FLASH_HBM` | 1 forces HBM scratch for comparison |
| `KUNLUN_FLASH_KV` | 1 enables row/feature parallel SET_ROWS; otherwise the existing kernel is retained |
| `KUNLUN_FLASH_TRACE` | Set only for serialized stage diagnostics; unset for normal performance runs |

The new KV writer uses parallel rows only after verifying a contiguous index sequence. It preserves row order for duplicate/irregular indices, uses the same FP16 rounding, and keeps the existing writer for unsupported layouts. Enable `KUNLUN_FLASH_KV=1` on both FA and separate-Attention runs to isolate FlashAttention's effect.

Build targets: `k200-flash-bench`, `k200-flash-contract`, and the existing Kunlun LLM/quality tools. The benchmark compares the three separate APIs with the complete Flash API, including preprocessing and completion; CPU reference and transfers are outside those timers. `KUNLUN_FLASH_BENCH_SAMPLE=1` samples four query rows for FP64 reference and must be disclosed when reporting numerical results. Stage tracing and LD_PRELOAD audit results are separate diagnostics.

State restore is validated within each mode. llama.cpp uses a different V cache layout with FA enabled; do not assume that states saved with FA disabled can be loaded into an FA-enabled context.

Next candidates are larger KQ residency in L1W/L1D, double-buffered L3 with verified stream/event dependencies, Flash-Decoding reduction, and lower online-softmax/mask overhead. Cross-kernel access to SDNN-private L1/L2 is not assumed.

## Hardware evaluation

See the [Japanese implementation and evaluation report](reports/flash-attention-20261026/REPORT.ja.md) and its raw summary tables for the measured comparison and limitations.

## 0.7.1 Online Softmax

`KUNLUN_FLASH_ONLINE=auto` (default) selects the optimized ordered-sum CL kernel; `legacy` retains the unmodified 0.7.0 kernel. See [Online Softmax](flash-online.md) for benchmarks, numerical guarantees and the Japanese evaluation report. The FlashAttention opt-in flags above remain required.

## 0.7.2 KQ reuse

`KUNLUN_FLASH_KQ=auto/legacy/reuse1024` controls Q/L1D and K/L1W reuse. See [KQ implementation and benchmarks](flash-kq.md). Online Softmax and all other attention components retain the 0.7.1 implementation.

## 0.7.3 Flash-Decoding

See [TG chunk execution, dispatch and benchmarks](flash-decoding.md). PP and the0.7.2 fallback remain unchanged.
