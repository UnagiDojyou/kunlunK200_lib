# Q4 GEMV address preparation opt-in

`KUNLUN_Q4_DMA_ADDR32=1` enables the bounded address-offset feeder for validated
Single/Layer shapes when the existing stream path is enabled. Unset, `0`, or any
other value preserves the previous feeder. `KUNLUN_Q4_MAC_DMA=0` also keeps the
previous path. `KUNLUN_Q4_NMAJOR` keeps its existing meaning.

Supported `(K,N)` pairs: `(4096,4096)`, `(4096,11008)`, `(11008,4096)`,
`(2048,256)`, `(2048,2048)`, `(2048,5632)`, `(5632,2048)`. Padded shapes and all
other shapes use the previous path. GEMM, Q4 ABI v1, packing, activation leases,
quantization, accumulation order, scale conversion, engine ownership, buffers,
DMA descriptors, fences, and barriers are unchanged.

The device ALU is 32-bit. Per-step signed 64-bit products used `__muldi3` in the
old feeder. The new feeder computes bounded nonnegative byte/element offsets
with 32-bit products, then uses the same wide global-pointer addition helper.
Global pointers are never truncated. The second weight bank address is formed
with one root-relative addition. The kernel rejects N > 11008, and keeps the
existing K <= 16384 bound; offsets stay far below INT_MAX. This is an address
calculation optimization, not a change to weight layout or transfer order.

`k200-q4-dma-addr32-bench PD K N M f32|f16 pattern trials` compares opt-in OFF/ON
for ordinary and prepared-activation APIs in alternating order, and validates
bitwise outputs, guard bytes, quantized activation, weight immutability, and
fallback on invalid settings. Existing `k200-common-q4-bench` supplies an
independent raw-Q4 CPU reference. The complete evaluation, raw commands, and
Japanese report are in `results/gemv-dmain1-20261102` in the development workspace.
No default was changed.
