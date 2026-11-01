GGML CPU reference benchmark
============================

The production library has no GGML dependency. ggml-reference.cpp is a validation-only comparison against the existing CPU GET_ROWS using a persistent six-thread GGML pool. Build with the library include path, ggml/include, libkunlunk200, libggml-cpu and libggml-base. Args: model.gguf absolute_tensor_offset K N PD timing(0/1). Use thermal guard/runner.

The independent public-API test is bench/get-rows-test.cpp (k200-get-rows-test target with KUNLUN_BUILD_BENCHMARKS=ON); its optional CPU timing is serial and is not the reported six-thread comparison. See ../../docs/get-rows/REPORT.ja.md.
