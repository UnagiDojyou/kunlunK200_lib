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

## Parallel RMS normalization

The FP32 RMS reduction uses 32 CL cores for partial sums, followed by a second kernel for normalization. Exact in-place inputs are preserved with the shared device copy kernel. Normal dispatch selects the parallel implementation for widths 512..16384 in multiples of 32 and at most 16 rows, where measured reduction savings exceed launch costs. Other shapes keep the existing implementation. `KUNLUN_RMS_NORM=legacy` forces the original path; `parallel` also evaluates eligible larger batches. `k200-rms-bench` reports both algorithms against double-precision CPU reductions, including zero, tiny and high-dynamic-range inputs and aliases. The legacy algorithm has a larger recorded error on wide high-dynamic-range inputs.

## Packed primary/residual FC candidate

`KUNLUN_GEMV=fused` submits primary and residual INT8 activation rows as a single doubled batch. One FC produces raw integer dot products; a CL kernel applies each component scale and the row weight scale. Both components and all row scales remain present. The original two-FC implementation stays available with `KUNLUN_GEMV=legacy` and remains the default pending complete model gates. Floating-point scaling order changes the result slightly. The fixed-corpus evaluation records the added error separately from the original rowwise INT8 approximation to Q4_0. `k200-fused-bench` compares the direct block-scaled Q4 implementation, two FCs and packed FC, including weight allocation bytes. Larger batches can favor the original path. Weight storage remains rowwise INT8; this candidate improves reuse and does not solve compressed-weight capacity.

## RMS parameter domain

Device RMS normalization requires finite positive epsilon. The GGML adapter leaves zero, negative and nonfinite epsilon to CPU evaluation; the direct low-level API rejects these parameters before launching a kernel. This avoids a device divide-by-zero for a zero row with zero epsilon. Model tests use the model-provided positive epsilon.

Device SET_ROWS accepts element-aligned, ordered non-overlapping destination row/head regions and disjoint input/index storage. The adapter routes other layouts to the CPU implementation. Read-only source/index strides may contain gaps or broadcasts. This conservative domain preserves sequential semantics for overlapping destinations.
