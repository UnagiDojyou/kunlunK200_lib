# Tiny M128 Q4 GEMM Owners opt-in

`KUNLUN_Q4_GEMM_TINY_OWNERS=1` extends the existing Owners dispatch to exact
`(K,N,M)` shapes `(2048,256,128)`, `(2048,2048,128)`, `(2048,5632,128)`,
and `(5632,2048,128)`. Unset, `0`, or any other value retains the old path.
The original input must match: padded M/K/N inputs do not opt in. Other input
M values, including a larger input split into M128 chunks, retain the old path.

`KUNLUN_Q4_GEMM_OWNERS` keeps its existing meaning. Its `0` or invalid values
disable all Owners dispatch, including this extension. The existing 7B shape
selection does not depend on the new variable. No kernel, weight ABI, arithmetic,
activation lease, buffer, or synchronization changes are made. Existing Rows128
Owners and Combo4 Owners images are reused. The new variable defaults to OFF.
