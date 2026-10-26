#include <kunlun/k200.h>
#include <xpu/runtime.h>
#include "flash-config.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace kunlun;
struct Arg{const void *p;size_t n;};
template<class T> Arg arg(const T&x){return {&x,sizeof(x)};}
static void launch(XPUFunc fn,std::initializer_list<Arg> args){size_t off=0;for(auto a:args){check(xpu_launch_argument_set(a.p,a.n,off));off+=a.n;}check(xpu_launch_config(4,16));check(xpu_launch_async(fn));}
static double ms(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char**argv){
 if(argc<9)return 2;
 int pd=atoi(argv[1]),d=atoi(argv[2]),block=atoi(argv[3]),m=atoi(argv[4]),h=atoi(argv[5]),mt=atoi(argv[6]),phase=atoi(argv[7]);
 if(pd<0||pd>1||(d!=64&&d!=128)||block<32||block>1024||block%32||m<1||m>256||h<1||h>128||phase<0||phase>3)return 2;
 std::string directory=argv[8];check(xpu_set_device(pd));int rows=m*h;
 KunlunFlashConfig p{};p.d=d;p.block=block;p.s=block*2;p.m=m;p.h=h;p.hk=h;p.b=p.bk=p.bv=1;p.scale=1.f/sqrtf(d);
 p.y[0]=4;p.y[1]=d*4;p.y[2]=d*h*4;p.y[3]=d*h*m*4;
 p.mask[0]=(mt==1||mt==7)?2:4;p.mask[1]=block*p.mask[0];p.mask[2]=p.mask[3]=0;p.mask_ne[2]=p.mask_ne[3]=1;p.mask_type=mt==1||mt==7;
 size_t sc=(size_t)rows*block*4,ou=(size_t)rows*d*4,st=(size_t)rows*32,total=sc+2*ou+st;
 if(total>16*1024*1024){fprintf(stderr,"standalone L3 workspace exceeds 16 MiB\n");return 2;}
 void *workspace=nullptr;check(xpu_malloc(&workspace,total,XPU_MEM_L3));require(workspace);
 void *score=workspace,*prob=score,*acc=(char*)workspace+sc,*partial=(char*)acc+ou,*stats=(char*)partial+ou;
 Buffer cfg(pd,256),mask(pd,(size_t)block*m*4),y(pd,ou),peaks(pd,(size_t)rows*32);
 check(copy(cfg.data(),&p,sizeof(p),host_to_device));
 std::vector<float> input(total/4),maskf((size_t)block*m);
 for(int row=0;row<rows;++row){for(int j=0;j<block;++j)input[(size_t)row*block+j]=sinf(j*.017f+row*.007f)*4;for(int j=0;j<d;++j){input[sc/4+(size_t)row*d+j]=sinf(j*.07f)*2;input[(sc+ou)/4+(size_t)row*d+j]=cosf(j*.09f);}
  input[(sc+2*ou)/4+row*8]=phase==2?100.f:0.2f;input[(sc+2*ou)/4+row*8+1]=100;
 }
 for(int q=0;q<m;++q)for(int j=0;j<block;++j)maskf[q*block+j]=(mt!=3&&j>block-m+q)?-INFINITY:0;
 if(mt==6)for(int row=0;row<std::min(rows,5);++row)for(int j=0;j<block;++j)input[(size_t)row*block+j]=row==0?NAN:(row==1?(j==0?INFINITY:1.f):(row==2?-INFINITY:(row==3?-10000.f:(j%2?-INFINITY:10000.f))));
 if(mt==4)for(auto&v:maskf)v=-INFINITY;
 if(mt==1||mt==7){std::vector<_Float16> half(maskf.size());for(size_t j=0;j<half.size();++j){half[j]=maskf[j];if(mt==7){unsigned short bits=j;memcpy(&half[j],&bits,2);}}check(copy(mask.data(),half.data(),half.size()*2,host_to_device));}
 else check(copy(mask.data(),maskf.data(),maskf.size()*4,host_to_device));check(wait());
 std::vector<float> reference;int failures=0;
 std::vector<std::string> modes={"", "-ordered"};if(argc>9)modes={argv[9]};
 for(const auto &suffix:modes){
  std::string base=directory+"/kunlun-flash-online"+suffix;
  std::ifstream bin(base+".bin",std::ios::binary),entry(base+".entry");std::vector<char>code((std::istreambuf_iterator<char>(bin)),{});std::string symbol;getline(entry,symbol);require(!code.empty()&&!symbol.empty());
  code.resize(((code.size()+255)&~size_t(255))+256);Buffer program(pd,code.size());check(copy(program.data(),code.data(),code.size(),host_to_device));check(wait());
  XPUFunc fn;check(xpu_create_cl_func(&fn,(uint64_t)(uintptr_t)program.data(),code.size(),0,0,0,symbol.c_str(),true));
  std::vector<double>timing,enqueue;int s0=phase?block:0,count=phase==3?0:block;
  const void *maskp=mt?mask.data():nullptr,*cp=cfg.data(),*qs=peaks.data();void *yp=y.data();
  for(int rep=0;rep<21;++rep){
   check(copy(workspace,input.data(),total,host_to_device));check(wait());
   auto t=std::chrono::steady_clock::now();launch(fn,{arg(yp),arg(prob),arg(acc),arg(stats),arg(partial),arg(score),arg(maskp),arg(qs),arg(cp),arg(s0),arg(count)});double enq=ms(t);check(wait());double elapsed=ms(t);if(rep){timing.push_back(elapsed);enqueue.push_back(enq);}
  }
  std::vector<float>out(total/4);check(copy(out.data(),workspace,total,device_to_host));check(wait());
  if(phase==3){out.resize(ou/4);check(copy(out.data(),y.data(),ou,device_to_host));check(wait());}
  double sq=0,en=0,maxerr=0;int bits=0,bad=0;
  if(reference.empty())reference=out;for(size_t i=0;i<out.size();++i){bits+=memcmp(&out[i],&reference[i],4)!=0;if(!std::isfinite(out[i])||!std::isfinite(reference[i])){if(!(std::isnan(out[i])&&std::isnan(reference[i]))&&out[i]!=reference[i])++bad;}else{double delta=out[i]-reference[i];sq+=delta*delta;en+=(double)reference[i]*reference[i];maxerr=std::max(maxerr,fabs(delta));}}
  std::sort(timing.begin(),timing.end());std::sort(enqueue.begin(),enqueue.end());
  printf("ONLINE_UNIT pd=%d D=%d block=%d M=%d H=%d mask=%d phase=%d mode=%s median_ms=%.6f min_ms=%.6f enqueue_ms=%.6f bits=%d nmse=%.12g max_abs=%.12g bad=%d workspace=%zu\n",pd,d,block,m,h,mt,phase,suffix.empty()?"legacy":suffix.c_str()+1,timing[timing.size()/2],timing[0],enqueue[enqueue.size()/2],bits,sq/std::max(en,1e-30),maxerr,bad,total);fflush(stdout);if(bad)++failures;
  if(suffix.find("profile")!=std::string::npos||suffix.find("split")!=std::string::npos){
   std::vector<unsigned int>ticks((size_t)rows*16);check(copy(ticks.data(),y.data(),ticks.size()*4,device_to_host));check(wait());
   for(int stage=0;stage<11;++stage){double sum=0;unsigned int mx=0;for(int r=0;r<rows;++r){sum+=ticks[r*16+stage];mx=std::max(mx,ticks[r*16+stage]);}printf("ONLINE_STAGE pd=%d M=%d D=%d block=%d mask=%d phase=%d mode=%s stage=%d mean_ticks=%.3f max_ticks=%u mean_worker_ms=%.6f\n",pd,m,d,block,mt,phase,suffix.c_str()+1,stage,sum/rows,mx,sum/900000./std::min(rows,64));}
  }
 }
 check(xpu_free(workspace));return failures?1:0;
}
