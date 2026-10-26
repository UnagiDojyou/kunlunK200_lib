#!/usr/bin/env bash
set -euo pipefail
XTDK="${XTDK_DIR:?set XTDK_DIR}"
SRC="${1:?source.xpu}"
OUT="${2:?outdir}"
name="$(basename "$SRC" .xpu)"
export LD_LIBRARY_PATH="$XTDK/shlib:$XTDK/lib:${LD_LIBRARY_PATH:-}"
xt() { "$XTDK/bin/$1" "${@:2}"; }
mkdir -p "$OUT/stubs/gnu"
: > "$OUT/stubs/gnu/stubs-x32.h"
inc="$(cd "$(dirname "$SRC")" && pwd)"
xt clang-10 --version
xt clang-10 -std=c++11 -O2 -fno-builtin -mcpu=xpu --xpu-device-only -c \
  -resource-dir "$XTDK/lib/clang/10.0.1" \
  -I"$XTDK/lib/clang/10.0.1/include" -I"$OUT/stubs" -I"$inc" \
  -o "$OUT/${name}.device.o" "$SRC"
# This CL kernel must fit its local stack before it can be loaded.
if [[ "$name" == kunlun-swiglu || "$name" == kunlun-attention-* || "$name" == kunlun-flash-* ]]; then
  xt clang-10 -std=c++11 -O2 -fno-builtin -mcpu=xpu --xpu-device-only -S \
    -resource-dir "$XTDK/lib/clang/10.0.1" \
    -I"$XTDK/lib/clang/10.0.1/include" -I"$OUT/stubs" -I"$inc" \
    -o "$OUT/${name}.device.s" "$SRC"
  awk '
    $1=="addi" && $2=="sp," && $3=="zero," { limit=$4 }
    $1=="addi" && $2=="sp," && $3=="sp," && $4<0 { if(-$4>frame)frame=-$4 }
    END {
      if(limit<=0 || frame<=0 || frame>limit) {
        printf "CL local frame unsafe or unknown: frame=%d limit=%d\n",frame,limit > "/dev/stderr"; exit 1
      }
      printf "Checked CL local frame: %d bytes (limit %d)\n",frame,limit
    }' "$OUT/${name}.device.s"
fi
entry="$(xt llvm-nm --defined-only -g "$OUT/${name}.device.o" | awk '$2=="T"{print $3}' | grep -v '^_start$' | head -n1)"
echo "$entry" > "$OUT/${name}.entry"
xt clang-10 -std=c++11 -O2 -fno-builtin -mcpu=xpu --xpu-device-only -c \
  -resource-dir "$XTDK/lib/clang/10.0.1" \
  -DKERNEL_ENTRY="$entry" -o "$OUT/${name}.crt.o" "$XTDK/bin/xpu-crt.xpu"
xt lld -flavor gnu -gc-sections "$OUT/${name}.crt.o" "$OUT/${name}.device.o" \
  "$XTDK/lib/linux/libclang_rt.builtins-xpu.a" -T "$XTDK/bin/xpu-kernel.t" \
  -o "$OUT/${name}.linked.elf"
xt llvm-objcopy "$OUT/${name}.linked.elf" --dump-section="KERNEL=$OUT/${name}.bin"
echo "wrote $OUT/${name}.bin entry=$entry"
