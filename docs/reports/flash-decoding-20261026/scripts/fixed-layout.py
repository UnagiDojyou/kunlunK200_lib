from pathlib import Path
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';P=R/'repos/llama.cpp/examples/kunlun'
p=L/'kernels/flash-decode-config.h';s=p.read_text().replace('int chunks;','int chunks; int chunk_keys; int reserved[7];');s+='\nstatic_assert(sizeof(FlashDecodeConfig)==192,"Flash-Decoding private metadata layout");\n';p.write_text(s)
old='int begin=(p.s/32*c/fd.chunks)*32,end=(p.s/32*(c+1)/fd.chunks)*32;'
new='int begin=fd.chunk_keys?c*fd.chunk_keys:(p.s/32*c/fd.chunks)*32,end=fd.chunk_keys?begin+fd.chunk_keys:(p.s/32*(c+1)/fd.chunks)*32;if(end>p.s)end=p.s;'
for p in (L/'kernels').glob('*decode*.xpu'):
 s=p.read_text()
 if old in s:p.write_text(s.replace(old,new))
p=L/'src/flash-decode.inc';s=p.read_text();s=s.replace('if(chunks>tiles)chunks=tiles;','int fixed=0;if(const char *raw=std::getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS")){int n=std::atoi(raw);if(n==32||n==64||n==128||n==256||n==512||n==1024)fixed=n;}\n if(fixed)chunks=(base.s+fixed-1)/fixed;\n if(chunks>tiles)chunks=tiles;')
s=s.replace('fd.p=base;fd.chunks=chunks;int maxkeys=((tiles+chunks-1)/chunks)*32;','fd.p=base;fd.chunks=chunks;fd.chunk_keys=fixed;int maxkeys=fixed?std::min(fixed,base.s):((tiles+chunks-1)/chunks)*32;')
s=s.replace('chunks=%d block=%d','chunks=%d fixed=%d block=%d').replace('base.hk,chunks,fd.p.block','base.hk,chunks,fixed,fd.p.block');p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('"${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-config.h"','"${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-config.h" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-decode-config.h"');p.write_text(s)
p=P/'flash-decoding-bench.cpp';s=p.read_text();needle='unsetenv("KUNLUN_FLASH_DECODING_WORK");if(mode=="head")'
s=s.replace(needle,'unsetenv("KUNLUN_FLASH_DECODING_FIXED_KEYS");if(mode=="fixed"||mode=="fixhead"){setenv("KUNLUN_FLASH_DECODING_FIXED_KEYS","1024",1);mode=mode=="fixhead"?"head":"on";}'+needle);p.write_text(s)
print('Added fixed-size KV chunks; balanced mode and public API/L3 layouts retained')
