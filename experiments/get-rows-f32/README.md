Validation only
===============

ggml-reference.cpp benchmarks the actual CPU F32 GET_ROWS using a persistent six-thread GGML pool and compares device row selection, transfer and complete-path costs. It is not part of the production library and introduces no GGML runtime dependency there. Args: K [PD]. Use the existing thermal guard and abort latch.

The independent public API/guard test is bench/get-rows-f32-test.cpp, available as k200-get-rows-f32-test when KUNLUN_BUILD_BENCHMARKS=ON. Args: [PD]. See ../../docs/get-rows-f32/REPORT.ja.md.
