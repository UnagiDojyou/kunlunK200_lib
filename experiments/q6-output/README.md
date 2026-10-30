Experimental only, rejected for performance. See REPORT.ja.md.
Compile with cmake/compile-native.sh experiments/q6-output/kunlun-q6-probe.xpu <outdir> and XTDK_DIR.
probe.cpp links the existing 0.7.12 library and ggml-cpu; argv: model offset K N image-directory. Run under the existing thermal guard.
