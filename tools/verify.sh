#!/usr/bin/env bash
set -euo pipefail
bench="${K200_BENCH:-k200-bench}"
guard="${K200_GUARD:-k200-thermal-guard}"
export KUNLUN_LOG_DIR="${KUNLUN_LOG_DIR:-$PWD/local-results}"
mkdir -p "$KUNLUN_LOG_DIR"
run() {
    local name="$1"; shift
    "$guard" --name "$name" --seconds 120 -- "$@" >"$KUNLUN_LOG_DIR/$name.txt" 2>&1 || { tail -30 "$KUNLUN_LOG_DIR/$name.txt"; return 1; }
    tail -3 "$KUNLUN_LOG_DIR/$name.txt"
}
for pd in 0 1; do
    run "library-memory-$pd" "$bench" memory f32 "$pd"
    for type in f32 f16 q4_0 i8; do run "library-matmul-$type-$pd" "$bench" matmul "$type" "$pd" 256 65 4; done
    for op in add multiply silu rms scale; do run "library-$op-$pd" "$bench" "$op" f32 "$pd"; done
done
run library-gemv-q4 "$bench" matmul q4_0 0 4096 4096 1
run library-gemv-i8 "$bench" matmul i8 0 4096 4096 1
