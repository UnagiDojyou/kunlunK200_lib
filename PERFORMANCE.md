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

## Indexed F16 cache updates

`set_rows_f16` writes indexed F32/F16 source values directly into strided F16 storage. Byte strides, I32/I64 indices, independent broadcast dimensions and IEEE round-to-nearest conversion are explicit in `SetRowsLayout`. All index ranges are checked on device before writing their elements. The GGML adapter validates indices and handles duplicates with the CPU reference. A validated contiguous transposed-cache pattern permits an eight-feature tile and token-run writes; the generic stride kernel remains available. `k200-set-rows-bench` compares entire destination storage, including untouched padding, against the compiler F16 reference.
