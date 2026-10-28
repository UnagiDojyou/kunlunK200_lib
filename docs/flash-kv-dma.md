# FlashAttention 0.7.6: pipeline and grouped K/V DMA

This release preserves the explicit backend FlashAttention opt-in. It adds the validated SDNN/CL pipeline, explicit bit-preserving `simd8` Online Softmax and grouped K/V transfers. KQ reuse1024, Wide PV, Flash-Decoding, tracked KV maxima, Q4 and SwiGLU remain available. See [pipeline ownership](flash-pipeline.md) and [Online modes](flash-online-extra.md).

## Grouped K/V transfers

Interleaved F16 cache is physically `[token][KV head][D]`. Grouping adjacent heads widens DMA rows; it does not reduce logical KV bytes or introduce a second cache layout. The SDNN converts F16 to symmetric INT16 and keeps visible F32 output. Quantization scales, key accumulation order and Online state equations are unchanged. Every API reads current inputs; no pointer-only KV reuse is added.

`KUNLUN_FLASH_KV_DMA=auto` is limited to measured D128/H32/HK32/B1/M1, Qt128, F16 mask without ALiBi, interleaved K/V, padded S512..4352 in steps of256, default block, and ordinary/pipeline FlashAttention. Flash-Decoding, PP and other shapes retain the previous transfers. `legacy` selects those transfers explicitly. `coalesced` enables supported HK>=32 group8 for PP/TG comparisons. `decode2/decode4` are explicit HK4 decoding experiments requiring at least four chunks. Raw `group2/group4/group8` tuning can regress and is not a general recommendation.

Unsupported layout/capacity, missing optional image or code-buffer allocation failure retain the previous transfer program. Explicit KQ legacy, PV narrow and decoding work overrides are respected. Each DMA payload fits its L2 bank; group8 D128/count1024 uses2 MiB of L1W per active SDNN. No hardware events, L3 semaphore, driver change or unverified DMA/MAC overlap is introduced.

## Measured benefit and limits

K200 KL1,900MHz, i7-8700, threads6, F16 KV, same Q4_0 GGUF and fixed generation history. At prefix4096, token-level ABBA with40 retained samples per mode gave Llama2 7B TG4.3712->5.6617 tok/s on PD0 (+29.52%) and4.4544->5.6962 on PD1 (+27.88%). Outputs were bitwise equal to the retained0.7.5 transfer path. This is prefix-dependent TG, not standard TG128. PP and TinyLlama benefits were not consistent, so grouped DMA is not automatically selected there. Explicit simd8 does not change Online auto (Ordered).

The subsequent DS/COA configuration/layout study did not establish an additional whole-model speedup. Its candidate shaders and failed layout probes are retained in the external experimental worktree, not promoted into this release. Version remains0.7.6. Extremely mixed query magnitudes in a shared quantization tile can retain pre-existing INT16 accuracy limits; bitwise old/new equality is not equivalence to CPU F64.

Standard llama-bench baseline0.7.6: PP512 and empty-KV TG128, batch512/ubatch128, ten samples each, F16 KV, threads6. Both models have CPU embedding and a CPU Q6_K output layer; these are mixed CPU/K200 measurements, not complete GPU offload. Layer and Row Split must be compared separately.

| Model | PD configuration | PP512 mean tok/s | TG128 mean tok/s |
| --- | --- | --- | --- |
| TinyLlama1.1B Q4_0 | PD0 |285.83|22.481|
| TinyLlama1.1B Q4_0 | PD1 |284.09|22.226|
| TinyLlama1.1B Q4_0 | Layer |280.97|22.627|
| TinyLlama1.1B Q4_0 | Row |173.88|13.310|
| Llama2 7B Q4_0 | PD0 |70.81|7.150|
| Llama2 7B Q4_0 | PD1 |70.81|7.325|
| Llama2 7B Q4_0 | Layer |70.65|7.060|
| Llama2 7B Q4_0 | Row |46.80|4.974|

The benchmark host commit was 558e95dc55b3414cc1fbfcf436ab0d5582cb5183. Tiny GGUF SHA256: da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556. 7B SHA256:78b8f9777dd620ad29cd2cffb6653b17fa8a5b1fddc1b8821180d60eedd24d48. Stock ubatch512 and prefix512+TG128 are separate measurements and must not be substituted for this table. No NVIDIA/AMD measurements were available.

## Build and comparison

Use the normal source build in README with external SDK paths. No generated image, model, SDK or runtime log is committed. New library-only API/unit comparisons are in `bench/flash-kv-dma-{unit,api,compat,final-api}.cpp`; they can be compiled against the installed public headers/library and XRE like other standalone benchmarks. Unit CLI is `PD D S_source M HK H shader_dir layout op [batch]`; it measures the final key tile rather than all S_source. API CLI is `PD D S M HK H [pattern] [batch]`; pattern29 uses interleaved KV and a F16 causal mask. `KV_DMA_CANDIDATE=auto` compares default against legacy. Use the existing thermal guard and stop latch for every device run.

For llama.cpp, use resident Attention and Flash KV opt-ins (`GGML_KUNLUN_ATTENTION=1`, `GGML_KUNLUN_FLASH=1`, `KUNLUN_FLASH_KV=1`). Keep common Q4, RoPE, SwiGLU, mask and cache settings identical between comparisons. The table used Online auto (Ordered), pipeline/decoding/max/DMA auto, common Q4 enabled, RoPE enabled and SwiGLU disabled. simd8 is an explicit separate comparison mode. Omit TRACE, VERIFY, PROFILE, device timestamp and profiling preload from normal timing. The supported llama-bench options and its empty-KV TG initialization must be checked when comparing other GPU backends.
