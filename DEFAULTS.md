# Recommended defaults (2026-11-01)

Unset environment variables now select the following configuration. Explicit 0 disables each newly enabled path; explicit 1 still respects all existing type, shape, stride, capacity, device and graph-reuse checks.

| Setting | Default |
|---|---|
| GGML_KUNLUN_COMMON_Q4 | 1 |
| GGML_KUNLUN_SWIGLU | 1 |
| KUNLUN_LARGE_PP | 0 |
| KUNLUN_Q6_K_SDNN | 1 |
| KUNLUN_Q6_K_LEGACY | 0 |
| GGML_KUNLUN_GET_ROWS | 1 |
| GGML_KUNLUN_GET_ROWS_F32 | 1 |
| GGML_KUNLUN_PRUNE_INACTIVE_MIXED | 1 |

Large PP remains opt-in. Legacy Q6 remains explicitly selectable with KUNLUN_Q6_K_LEGACY=1 while SDNN is enabled; KUNLUN_Q6_K_SDNN=0 restores CPU Q6 fallback. COMMON_Q4=0 restores the previous Q4 selection priority. GET_ROWS=0 and GET_ROWS_F32=0 independently restore CPU gathers. PRUNE_INACTIVE_MIXED=0 restores the original graph builder.

SwiGLU, common-Q4 activation quantization and SDNN Q6 can change numerical output relative to older CPU/CDNN configurations. The previous reports document these differences. GET_ROWS and graph pruning matched their prior paths bitwise in the tested models. Tiny TG measurements for graph pruning were variable and had a lower mean; this requested default policy is not a claim of universal speedup. Unsupported models/inputs retain existing fallback and mixed-input reconstruction. Kernel algorithms, weight ABI, leases, safety guards and existing explicit settings are unchanged.

These defaults supersede historical opt-in/default-OFF descriptions in dated reports. See the source and latest configuration table for current behavior.
