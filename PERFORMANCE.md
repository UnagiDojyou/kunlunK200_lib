# Device copies and in-place elementwise operations

`copy_rows` copies contiguous F32 rows to a strided F32 destination on one PD.
Stride and column offset are measured in F32 elements. Source and destination
storage must not overlap. The caller owns storage and must wait for completion
before reusing it. A kernel handle belongs to one PD.

The kernel uses bounded aligned local tiles. It does not allocate device memory
per call and does not invoke the runtime D2D API for each output row.

In-place elementwise operations preserve any aliased input before launching the
arithmetic kernel. Repeated multi-row in-place MUL was observed to leave some
output elements unchanged in the previous implementation. The underlying
compiler/runtime cause has not been isolated. Staging the input removes the
observed error. `KUNLUN_ALIAS_COPY=legacy` selects runtime D2D input staging for
comparison; the default uses the device copy kernel. Both paths preserve inputs.

Build `k200-alias-bench` and `k200-data-bench` for independent CPU comparisons.
The alias benchmark checks 100 repetitions, both input aliases, full-size inputs
and vector broadcasting. The copy benchmark checks every output value and guard
region, then reports the median and range of five groups of 20 completed copies.
