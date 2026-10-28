# Opt-in KL1 Online Softmax for FlashAttention 0.7.5

`KUNLUN_FLASH_ONLINE=simd8` selects scalar native `max_s`, SIMD8 packed
F16 mask expansion, and bitwise SIMD +0 fills. Native exp and the legacy
left-to-right FP32 sum order are retained. Signed-zero and nonfinite max
cases preserve the strict legacy comparison result, replaying from st[0]
where necessary. No quantization or SDNN kernel is changed.

The default auto, legacy and ordered modes retain their previous behavior.
unroll16 is an intermediate comparison mode, not automatically selected.
Short-context / TG model gains were mixed; do not infer a universal speedup.

Normal and existing pipeline FlashAttention use the new regular kernel.
Flash-Decoding uses independently tested head/chunk Online kernels for
explicit simd8. Existing Decoding/pipeline shape selection and fallbacks
remain unchanged. L3/workspace, Score/P alias and acc/partial/stats ABI are
unchanged; stats reserves eight floats but only m/l are defined normal state.

Existing API timestamp profiles remain legacy Ordered; use the separately
compiled Online diagnostic benchmark for simd8 stage timings. Timestamp
code changes scheduling and is not an additive constant overhead. Normal
token/s must not enable trace, profile or verification.

See the task's REPORT.ja.md and REPRODUCE.ja.md for numerical, model,
compatibility, worker/SFU, memory and repeated-comparison results.
