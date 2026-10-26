from pathlib import Path
import subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-decoding-20261026'
p=L/'src/flash-decode.inc';s=p.read_text().replace('const char *number=','int power2=1;while(power2<chunks)power2*=2;chunks=power2;\n const char *number=')
s=s.replace('bool head=std::getenv("KUNLUN_FLASH_DECODING_LAYOUT")&&!std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_LAYOUT"),"head");','bool head=!std::getenv("KUNLUN_FLASH_DECODING_LAYOUT")||std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_LAYOUT"),"chunk");')
p.write_text(s)
p=L/'bench/flash-decode-bench.cpp';s=p.read_text().replace('#include <kunlun/k200.h>','#include <kunlun/k200.h>\n#include <xpu/runtime.h>')
s=s.replace(' auto qp=dense',' void *reserved_l3=nullptr;if(auto hold=getenv("KUNLUN_FLASH_DECODE_BENCH_RESERVE_L3_MB")){int mb=atoi(hold);if(mb<1||mb>14)return 2;check(xpu_set_device(pd));check(xpu_malloc(&reserved_l3,(size_t)mb*1024*1024,XPU_MEM_L3));}\n auto qp=dense',1)
s=s.replace('if(pattern==12){maskp.type','if(pattern==12||pattern==25){maskp.type')
s=s.replace('if(pattern==7)p.max_bias=8;','if(pattern==7||pattern==24||pattern==25)p.max_bias=8;')
marker=' auto qr=q.raw,kr=k.raw,vr=v.raw,mr=mask.raw;'
extra=''' if(pattern==24||pattern==25){for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>s-m+r?-INFINITY:-(s-1-x)*.01f,x,r);}
 if(pattern==27||pattern==28){for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put((pattern==27?(x>=s/4&&x<s/2):(x>=s/2))?-INFINITY:0.f,x,r);}
'''
s=s.replace(marker,extra+marker).replace(' return pass?0:1;',' if(reserved_l3)check(xpu_free(reserved_l3));\n return pass?0:1;');p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('target_link_libraries(k200-flash-decode-bench PRIVATE KunlunK200::kunlunk200)','target_include_directories(k200-flash-decode-bench PRIVATE "${KUNLUN_RUNTIME_INCLUDE}")\n    target_link_libraries(k200-flash-decode-bench PRIVATE KunlunK200::kunlunk200 "${KUNLUN_XPURT}")');p.write_text(s)
ids=(O/'inputs/prompt2048.ids').read_text().split();(O/'inputs/prompt4096.ids').write_text(' '.join(ids*2)+'\n')
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
for build,targets in [('flash-decode-lib',['k200-flash-decode-bench']),('flash-decode-llama',['kunlun-llm-smoke','kunlun-quality-eval','kunlun-attention-test','kunlun-swiglu-test','kunlun-backend-contract'])]:subprocess.run(['cmake','--build',str(R/'build'/build),'-j6','--target',*targets],env=env,check=True)
