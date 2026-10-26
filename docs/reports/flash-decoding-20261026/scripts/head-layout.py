from pathlib import Path
import subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200'
s=(L/'kernels/kunlun-flash-decode-online.xpu').read_text().replace('kunlun_flash_decode_online(','kunlun_flash_decode_online_head(').replace('int c=storage/rows,row=storage%rows;','int c=storage/rows,row=storage%rows,state=row*fd.chunks+c;')
for name,stride in [('acc','p.d'),('partial','p.d'),('stats','8')]:s=s.replace(name+'+storage*'+stride,name+'+state*'+stride)
(L/'kernels/kunlun-flash-decode-online-head.xpu').write_text(s)
s=(L/'kernels/kunlun-flash-decode-pv.xpu').read_text().replace('kunlun_flash_decode_pv(','kunlun_flash_decode_pv_head(').replace('out+(c*p.h*p.b+(b*p.h+h))*p.d','out+((b*p.h+h)*fd.chunks+c)*p.d').replace('dma_o(lowptr(dst),e,rows*p.d*4);','dma_cfg_2d(rows,fd.chunks*p.d*4,p.d*4);dma_o_2d(lowptr(dst),e,p.d*4);')
(L/'kernels/kunlun-flash-decode-pv-head.xpu').write_text(s)
(L/'kernels/kunlun-flash-decode-reduce-head.xpu').write_text(r'''#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/simd.h"
#include "xpu/kernel/math.h"
#include "flash-decode-config.h"
__global__ void kunlun_flash_decode_reduce_head(unsigned char *y,const float *acc,const float *partial,const float *stats,const FlashDecodeConfig *cfg){
 __local__ FlashDecodeConfig fd;GM2LM(cfg,&fd,sizeof(fd));
 __local__ float a[1024],part[1024],sum[128],st[512];
 int rows=fd.p.h*fd.p.b,d=fd.p.d,tid=core_id()*cluster_num()+cluster_id(),workers=core_num()*cluster_num();
 for(int row=tid;row<rows;row+=workers){
  GM2LM(stats+row*fd.chunks*8,st,fd.chunks*32);float maximum=-__builtin_inff();
  for(int c=0;c<fd.chunks;++c)if(st[c*8]>maximum)maximum=st[c*8];
  for(int j=0;j<d;++j)sum[j]=0;float denom=0;
  if(maximum!=-__builtin_inff())for(int first=0;first<fd.chunks;first+=8){
   int n=fd.chunks-first;if(n>8)n=8;
   GM2LM(acc+(row*fd.chunks+first)*d,a,n*d*4);GM2LM(partial+(row*fd.chunks+first)*d,part,n*d*4);
   for(int c=0;c<n;++c){int i=first+c;if(st[i*8+1]<=0||st[i*8]==-__builtin_inff())continue;
    float weight=st[i*8]==maximum?1.0f:expf(st[i*8]-maximum);denom+=weight*st[i*8+1];
    for(int j=0;j<d;j+=8){_x256_vvadd_ls(a+c*d+j,part+c*d+j,a+c*d+j);if(weight!=1)_x256_svmul_ls(weight,a+c*d+j,a+c*d+j);_x256_vvadd_ls(sum+j,a+c*d+j,sum+j);}
   }
  }
  float inv=denom>0?1.0f/denom:0;for(int j=0;j<d;j+=8)_x256_svmul_ls(inv,sum+j,sum+j);
  int h=row%fd.p.h,b=row/fd.p.h;LM2GM(sum,y+h*fd.p.y[1]+b*fd.p.y[3],d*4);
 }
}
''')
p=L/'src/flash.inc';s=p.read_text().replace('decode_reduce,qpack','decode_reduce,decode_pv_head,decode_online_head,decode_reduce_head,qpack').replace('&decode_reduce,&qpack','&decode_reduce,&decode_pv_head,&decode_online_head,&decode_reduce_head,&qpack');p.write_text(s)
p=L/'src/flash-decode.inc';s=p.read_text();start=s.index(' load_kernel(s.decode_kq,');end=s.index(' bool trace=',start)
s=s[:start]+''' bool head=std::getenv("KUNLUN_FLASH_DECODING_LAYOUT")&&!std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_LAYOUT"),"head");
 auto &online=head?s.decode_online_head:s.decode_online;auto &pv=head?s.decode_pv_head:s.decode_pv;auto &reduce=head?s.decode_reduce_head:s.decode_reduce;
 load_kernel(s.decode_kq,"kunlun-flash-decode-kq",id,true);load_kernel(pv,head?"kunlun-flash-decode-pv-head":"kunlun-flash-decode-pv",id,true);load_kernel(online,head?"kunlun-flash-decode-online-head":"kunlun-flash-decode-online",id,false);load_kernel(reduce,head?"kunlun-flash-decode-reduce-head":"kunlun-flash-decode-reduce",id,false);
'''+s[end:]
s=s.replace('native_launch(s.decode_online,','native_launch(online,').replace('native_launch(s.decode_pv,','native_launch(pv,').replace('native_launch(s.decode_reduce,','native_launch(reduce,').replace('storage=%s calls=%zu','storage=%s layout=%s calls=%zu').replace('hbm?"HBM":"L3",s.calls','hbm?"HBM":"L3",head?"head":"chunk",s.calls');p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('foreach(name kunlun-flash-decode-kq','foreach(name kunlun-flash-decode-online-head kunlun-flash-decode-pv-head kunlun-flash-decode-reduce-head kunlun-flash-decode-kq');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench'],env=env,check=True)
