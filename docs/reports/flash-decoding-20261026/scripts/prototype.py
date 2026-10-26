from pathlib import Path
import subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-decoding-20261026'
(L/'kernels/flash-decode-config.h').write_text('#pragma once\n#include "flash-config.h"\nstruct FlashDecodeConfig {KunlunFlashConfig p;int chunks;};\n')
def init(s,name):
 s=s.replace('#include "flash-config.h"','#include "flash-decode-config.h"')
 s=s.replace('int s0,int count){','int step){')
 s=s.replace('__local__ KunlunFlashConfig p;GM2LM(cfg,&p,sizeof(p));','__local__ FlashDecodeConfig fd;GM2LM(cfg,&fd,sizeof(fd));__local__ KunlunFlashConfig p;p=fd.p;')
 s=s.replace('const KunlunFlashConfig *cfg','const FlashDecodeConfig *cfg')
 return s
s=init((L/'kernels/flash-kq-device.h').read_text(),'kq').replace('KQ_NAME','kunlun_flash_decode_kq').replace('KQ_OUTPUT_KEYS','1024')
start=s.index(' int nt=count/32,');end=s.index('  // Each input DMA',start)
s=s[:start]+''' int mt=1,groups=p.h/p.hk,units=p.b*p.hk*fd.chunks;
 bool grouped=groups>1&&p.tg_group;
 int maxparts=(cluster_num()+units-1)/units;if(maxparts>p.block/32)maxparts=p.block/32;
 for(int task=cluster_id();task<units*maxparts;task+=cluster_num()){
  int part=task%maxparts,unit=task/maxparts,c=unit%fd.chunks,hk=unit/fd.chunks%p.hk,b=unit/fd.chunks/p.hk;
  int begin=(p.s/32*c/fd.chunks)*32,end=(p.s/32*(c+1)/fd.chunks)*32;
  int s0=begin+step*p.block,count=end-s0;if(count>p.block)count=p.block;if(count<=0)continue;
  int nt=count/32,parts=maxparts;if(parts>nt)parts=nt;if(part>=parts)continue;
  int first=(part*nt/parts)*32,nkeys=((part+1)*nt/parts)*32-first;
'''+s[end:]
s=s.replace('score+((b*p.h+h)*p.m+m0)*p.block','score+(c*p.h*p.b+(b*p.h+h))*p.block')
(L/'kernels/kunlun-flash-decode-kq.xpu').write_text(s)
s=init((L/'kernels/kunlun-flash-pv-wide.xpu').read_text(),'pv').replace('kunlun_flash_pv_wide','kunlun_flash_decode_pv')
s=s.replace('mt=(p.m+p.qt-1)/p.qt','mt=1')
s=s.replace('for(int task=cluster_id();task<p.b*p.hk;task+=cluster_num()){\n  int hk=task%p.hk,b=task/p.hk;','''for(int task=cluster_id();task<p.b*p.hk*fd.chunks;task+=cluster_num()){
  int c=task%fd.chunks,hk=task/fd.chunks%p.hk,b=task/fd.chunks/p.hk;
  int begin=(p.s/32*c/fd.chunks)*32,end=(p.s/32*(c+1)/fd.chunks)*32;
  int s0=begin+step*p.block,count=end-s0;if(count>p.block)count=p.block;if(count<=0)continue;''')
s=s.replace('prob+((b*p.h+h)*p.m+m0)*p.block','prob+(c*p.h*p.b+(b*p.h+h))*p.block').replace('out+((b*p.h+h)*p.m+m0)*p.d','out+(c*p.h*p.b+(b*p.h+h))*p.d')
(L/'kernels/kunlun-flash-decode-pv.xpu').write_text(s)
s=(L/'kernels/flash-online-device.h').read_text();s=init(s,'online').replace('ONLINE_NAME','kunlun_flash_decode_online')
s='#define ONLINE_REDUCE 2\n#define ONLINE_REUSE 0\n'+s
start=s.index(' int cached_off=-1;');end=s.index('  int q=row%p.m,',start)
s=s[:start]+''' int cached_off=-1;
 for(int storage=tid;storage<rows*fd.chunks;storage+=step_workers){
  int c=storage/rows,row=storage%rows;
  int begin=(p.s/32*c/fd.chunks)*32,end=(p.s/32*(c+1)/fd.chunks)*32;
  int s0=begin+step*p.block,count=end-s0;if(count>p.block)count=p.block;if(count<=0)continue;
'''+s[end:]
# The existing worker-stride variable is independent of the wave argument.
s=s.replace('step=cluster_num()*core_num(),rows','step_workers=cluster_num()*core_num(),rows')
s=s.replace('if(s0){GM2LM','if(step){GM2LM')
for prefix,mult in [('acc','p.d'),('partial','p.d'),('stats','8'),('score','p.block'),('prob','p.block')]:s=s.replace(prefix+'+row*'+mult,prefix+'+storage*'+mult)
(L/'kernels/kunlun-flash-decode-online.xpu').write_text(s)
(L/'kernels/kunlun-flash-decode-reduce.xpu').write_text(r'''#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/simd.h"
#include "xpu/kernel/math.h"
#include "flash-decode-config.h"
__global__ void kunlun_flash_decode_reduce(unsigned char *y,const float *acc,const float *partial,const float *stats,const FlashDecodeConfig *cfg){
 __local__ FlashDecodeConfig fd;GM2LM(cfg,&fd,sizeof(fd));
 __local__ float a[128],part[128],sum[128],st[8],maxima[64],denoms[64];
 int rows=fd.p.h*fd.p.b,d=fd.p.d,tid=core_id()*cluster_num()+cluster_id(),workers=core_num()*cluster_num();
 for(int row=tid;row<rows;row+=workers){
  float maximum=-__builtin_inff();
  for(int c=0;c<fd.chunks;++c){GM2LM(stats+(c*rows+row)*8,st,32);maxima[c]=st[0];denoms[c]=st[1];if(st[0]>maximum)maximum=st[0];}
  for(int j=0;j<d;++j)sum[j]=0;float denom=0;
  if(maximum!=-__builtin_inff())for(int c=0;c<fd.chunks;++c){
   if(denoms[c]<=0||maxima[c]==-__builtin_inff())continue;
   float weight=maxima[c]==maximum?1.0f:expf(maxima[c]-maximum);denom+=weight*denoms[c];
   GM2LM(acc+(c*rows+row)*d,a,d*4);GM2LM(partial+(c*rows+row)*d,part,d*4);
   for(int j=0;j<d;j+=8){_x256_vvadd_ls(a+j,part+j,a+j);if(weight!=1)_x256_svmul_ls(weight,a+j,a+j);_x256_vvadd_ls(sum+j,a+j,sum+j);}
  }
  float inv=denom>0?1.0f/denom:0;for(int j=0;j<d;j+=8)_x256_svmul_ls(inv,sum+j,sum+j);
  int h=row%fd.p.h,b=row/fd.p.h;LM2GM(sum,y+h*fd.p.y[1]+b*fd.p.y[3],d*4);
 }
}
''')
# Isolate host FD logic and retain the old PP/TG path verbatim as fallback.
(L/'src/flash-decode.inc').write_text(r'''#include "../kernels/flash-decode-config.h"
namespace {
bool flash_decode(FlashState &s,int id,void *y,const void *q,const void *k,const void *v,const void *mask,const KunlunFlashConfig &base){
 const char *mode=std::getenv("KUNLUN_FLASH_DECODING");
 if(base.m!=1||!mode||std::strcmp(mode,"on"))return false;
 int tiles=base.s/32,chunks=(base.s+1023)/1024;
 const char *length=std::getenv("KUNLUN_FLASH_DECODING_CHUNK");
 if(length){int n=std::atoi(length);if(n==256||n==512||n==1024)chunks=(base.s+n-1)/n;}
 const char *number=std::getenv("KUNLUN_FLASH_DECODING_CHUNKS");
 if(number){int n=std::atoi(number);if(n==1||n==2||n==4||n==8||n==16||n==32||n==64)chunks=n;}
 if(chunks>tiles)chunks=tiles;if(chunks<1||chunks>64)return false;
 FlashDecodeConfig fd{};fd.p=base;fd.chunks=chunks;int maxkeys=((tiles+chunks-1)/chunks)*32;fd.p.block=std::min(base.block,maxkeys);
 size_t rows=(size_t)base.h*base.b*chunks,sc=rows*fd.p.block*4,ou=rows*base.d*4,total=sc+2*ou+rows*32;
 if(total>16*1024*1024)return false;
 if(!s.l3){s.l3=native_l3[id].lock();if(!s.l3){s.l3=std::make_shared<NativeL3>(id);native_l3[id]=s.l3;}}
 bool hbm=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");void *workspace=nullptr;
 if(hbm){native_reserve(s.hbm,id,total);workspace=s.hbm->data();}else if(s.l3->reserve(total))workspace=s.l3->pointer;else return false;
 native_reserve(s.config,id,256);if(s.config_cache.size()!=sizeof(fd)||std::memcmp(s.config_cache.data(),&fd,sizeof(fd))){check(xpu_memcpy(s.config->data(),&fd,sizeof(fd),XPU_HOST_TO_DEVICE));check(xpu_wait());s.config_cache.assign((unsigned char*)&fd,(unsigned char*)&fd+sizeof(fd));}
 s.workspace=total;s.last_l3=!hbm;s.last_block=fd.p.block;s.last_qt=base.qt;
 void *score=workspace,*prob=score,*partial=(char*)score+sc,*acc=(char*)partial+ou,*stats=(char*)acc+ou,*cp=s.config->data(),*kp=s.peaks->data(),*qr=s.ranges->data(),*vp=(char*)kp+256;
 load_kernel(s.decode_kq,"kunlun-flash-decode-kq",id,true);load_kernel(s.decode_pv,"kunlun-flash-decode-pv",id,true);load_kernel(s.decode_online,"kunlun-flash-decode-online",id,false);load_kernel(s.decode_reduce,"kunlun-flash-decode-reduce",id,false);
 bool trace=std::getenv("KUNLUN_FLASH_TRACE");auto tick=std::chrono::steady_clock::now();
 auto stage=[&](const char *name){if(trace){check(xpu_wait());auto now=std::chrono::steady_clock::now();std::fprintf(stderr,"DECODE_STAGE stage=%s ms=%.6f\n",name,std::chrono::duration<double,std::milli>(now-tick).count());tick=now;}};
 int waves=(maxkeys+fd.p.block-1)/fd.p.block;
 for(int step=0;step<waves;++step){int unused=0;
  native_launch(s.decode_kq,{native_arg(score),native_arg(k),native_arg(q),native_arg(kp),native_arg(qr),native_arg(cp),native_arg(step)},8);stage("kq");
  native_launch(s.decode_online,{native_arg(y),native_arg(prob),native_arg(acc),native_arg(stats),native_arg(partial),native_arg(score),native_arg(mask),native_arg(qr),native_arg(cp),native_arg(step)},16);stage("online");
  native_launch(s.decode_pv,{native_arg(partial),native_arg(v),native_arg(prob),native_arg(vp),native_arg(cp),native_arg(step)},8);stage("pv");
 }
 native_launch(s.decode_reduce,{native_arg(y),native_arg(acc),native_arg(partial),native_arg(stats),native_arg(cp)},16);check(xpu_wait());stage("reduce");
 if(!s.calls++||trace)std::fprintf(stderr,"KUNLUN Flash-Decoding PD%d D=%d S=%d H=%d HK=%d chunks=%d block=%d waves=%d workspace=%zu storage=%s calls=%zu\n",id,base.d,base.s,base.h,base.hk,chunks,fd.p.block,waves,total,hbm?"HBM":"L3",s.calls);
 return true;
}
}
''')
p=L/'src/flash.inc';s=p.read_text().replace('online_ordered,qpack','online_ordered,decode_kq,decode_pv,decode_online,decode_reduce,qpack').replace('&online_ordered,&qpack','&online_ordered,&decode_kq,&decode_pv,&decode_online,&decode_reduce,&qpack')
s=s.replace('const char *flash_attention_reason', '#include "flash-decode.inc"\nconst char *flash_attention_reason',1)
needle='    native_reserve(s.config,id,256);'
s=s.replace(needle,'    if(flash_decode(s,id,y,q,k,v,p.has_mask?mask:nullptr,cfg))return true;\n'+needle,1);p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('foreach(name kunlun-flash-kq-reuse1024','foreach(name kunlun-flash-decode-kq kunlun-flash-decode-pv kunlun-flash-decode-online kunlun-flash-decode-reduce kunlun-flash-kq-reuse1024').replace('"${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-kq-device.h")','"${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-kq-device.h" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash-decode-config.h")')
s=s.replace('    add_executable(k200-flash-kq-api-bench','    add_executable(k200-flash-decode-bench bench/flash-decode-bench.cpp)\n    target_link_libraries(k200-flash-decode-bench PRIVATE KunlunK200::kunlunk200)\n    add_executable(k200-flash-kq-api-bench');p.write_text(s)
s=(L/'bench/flash-kq-api-bench.cpp').read_text().replace('KUNLUN_FLASH_KQ','KUNLUN_FLASH_DECODING').replace('KQ_COMPARE','DECODE_COMPARE').replace('"legacy",1','"off",1')
s=s.replace('bool pass=bitdiff==0&&!bad','bool pass=!bad')
(L/'bench/flash-decode-bench.cpp').write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','-S',str(L),'-B',str(R/'build/flash-decode-lib'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'],env=env,check=True)
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench'],env=env,check=True)
