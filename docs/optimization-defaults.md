# Optimization defaults (KunlunK200 0.7.12)

The common-Q4 execution/quantization profile is still opt-in. These defaults do not enable common Q4, Attention, or FlashAttention by themselves. Set options before creating a model/context and keep them fixed for its lifetime.

| Setting | Unset in 0.7.11 | Unset in 0.7.12 | Scope / dependency |
| --- | --- | --- | --- |
| GGML_KUNLUN_Q4_ACT_SHARE | off | on when COMMON_Q4=1 | Backend only: contiguous F32 matrix, K<=16384, same tensor/graph/group generation, dtype/shape/strides and owning PD; at least two consumers; explicit Lease and 16MiB preparation limit |
| KUNLUN_Q4_MAC_DMA | off | on for validated shapes | ABI v1 common Q4, M=1, unpadded K, validated padded N; existing L3 capacity/allocation guard |
| KUNLUN_Q4_NMAJOR | off | on for validated shapes | Requires stream selection; Kp/Np=4096/4096,4096/11008,11008/4096 |
| KUNLUN_Q4_GEMM_OWNERS | off | on for validated shapes | Physical M=tileM=128, Kp/Np=4096/4096,4096/11008,11008/4096; original Rows/Combo capacity checks |
| KUNLUN_LARGE_PP | off | off | Explicit 1 retains 0.7.11 bounded-query Flash, RoPE/CONT placement and KV-writer chunks. Numerical CPU/XPU differences remain quality-sensitive |
| GGML_KUNLUN_SWIGLU | off | off | Explicit 1 runs supported PP/TG GLU on XPU; pp restricts to multi-row GLU. CPU SIMD / CL exp rounding can cross A8 quantization thresholds |

For the four newly automatic options, explicit 0 restores the preceding path, 1 enables selection within exactly the existing guards, and other values disable the feature (with bounded diagnostics). MAC_DMA=0 also prevents N-major selection. GEMM Owners and sharing do not depend on stream selection. No weight repack, ABI, quantizer, accumulation order, fences or kernel changed.

Stream Kp/Np whitelist: 2048/{256,1024,2048,2816,5632}, 4096/{256,2048,4096,5632,11008}, 5632/{1024,2048}, 11008/{2048,4096}; K must equal Kp. Nonmatching shapes retain the preceding native/common path. Row Split applies these guards to each owning PD's partition. Unsupported prepared paths retain ordinary common Q4; stale/wrong-PD/written/expired leases are not reused. The preparation-size and L3 fallbacks remain. A runtime allocation/kernel error keeps the existing fail-stop behavior; the defaults do not authorize continuing after a device error or promise recovery from total HBM exhaustion.

GGML_KUNLUN_DIAGNOSTICS=1 prints one startup policy line and existing bounded placement/shape diagnostics. Selection counters show actual execution; configured enablement alone does not prove a kernel was selected. Standalone library stream/N-major/Owners first-use diagnostics remain.

## Other existing options (unchanged)

- COMMON_Q4 unset/0: off; 1: common ABI v1. Precedence remains common > legacy NATIVE_Q4 > REQUANT_I8 > direct Q4. Selecting common changes the numerical profile (Row A8 for M>1), and was deliberately not made automatic.
- NATIVE_Q4 unset: off. REQUANT_I8 unset: off; existing nonzero-string semantics unchanged.
- ATTENTION unset: off; 1 or tg opt in. ROPE defaults on only inside enabled Attention. FLASH requires explicit 1 as well as enabled Attention and supported layouts. CLI Flash enablement is also required.
- KUNLUN_FLASH_KV and KUNLUN_BROADCAST_VECTOR require explicit 1. They remain independent, unchanged options.
- RMS_NORM retains its existing shape-based parallel selection; explicit legacy/parallel overrides remain. ROW_COPY/ROW_SPLIT/SET_ROWS policies are unchanged.
- Flash Online/KQ/Decoding/Pipeline/KV_DMA/KV_MAX and attention preparation retain their current auto policies, dependent on an enabled Attention/Flash path. Diagnostic/profile/custom-tile and HBM overrides remain diagnostic/experimental options, not newly enabled defaults.

## Example

Keep the stable common-Q4 + Attention profile, remove the four redundant flags:

~~~sh
export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1
export GGML_KUNLUN_ROPE=1 KUNLUN_FLASH_KV=1 KUNLUN_BROADCAST_VECTOR=1
unset GGML_KUNLUN_Q4_ACT_SHARE KUNLUN_Q4_MAC_DMA KUNLUN_Q4_NMAJOR KUNLUN_Q4_GEMM_OWNERS
# Optional, independently evaluated numerical modes:
# export KUNLUN_LARGE_PP=1 GGML_KUNLUN_SWIGLU=1
# Previous ordinary common-Q4 behavior:
# export GGML_KUNLUN_Q4_ACT_SHARE=0 KUNLUN_Q4_MAC_DMA=0 KUNLUN_Q4_NMAJOR=0 KUNLUN_Q4_GEMM_OWNERS=0
# Explicit previous optimized configuration: set those four variables to 1.
~~~

The new system-library backend requires 0.7.12, preventing a new backend from silently using old opt-in library defaults. Bundled builds use the pinned submodule. CPU-only, CDNN and direct/legacy Q4 settings keep their existing precedence and behavior.
