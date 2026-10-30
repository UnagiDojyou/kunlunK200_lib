#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <array>
#include <cstdio>
#include <cstdlib>
using namespace native_q4_bench;
static void verify(bool b){if(!b){std::fprintf(stderr,"Q4_DEFAULTS_FAIL\n");std::exit(2);}}
static std::vector<unsigned char> get(const void*p,size_t n){std::vector<unsigned char>v(n);kunlun::check(kunlun::copy(v.data(),p,n,kunlun::device_to_host));kunlun::check(kunlun::wait());return v;}
int main(int argc,char**argv){int pd=argc>1?atoi(argv[1]):0;verify(pd==0||pd==1);
 for(auto shape:{std::array<int,3>{4096,4096,1},{2048,2048,1},{4096,4096,128},{512,512,1},{512,512,32}}){
  int k=shape[0],n=shape[1],m=shape[2];Weights raw(k,n);auto x=input(k,m,0);kunlun::CommonQ4Weight w(pd,k,n);w.upload(raw.q4.data(),raw.q4.size());kunlun::Buffer dx(pd,x.size()*4),dy(pd,(size_t)m*n*4);
  kunlun::check(kunlun::copy(dx.data(),x.data(),x.size()*4,kunlun::host_to_device));kunlun::check(kunlun::wait());std::vector<unsigned char>gold;
  for(const char *mode:{"0","1","unset","invalid"}){
   for(const char*name:{"KUNLUN_Q4_MAC_DMA","KUNLUN_Q4_NMAJOR","KUNLUN_Q4_GEMM_OWNERS"})if(!strcmp(mode,"unset"))unsetenv(name);else setenv(name,mode,1);
   kunlun::CommonQ4Workspace ws;kunlun::common_q4_matmul(ws,pd,dy.data(),w,dx.data(),m,kunlun::InputType::f32);auto y=get(dy.data(),(size_t)m*n*4);if(gold.empty())gold=y;else verify(gold==y);
   auto stats=ws.mac_statistics();bool on=!strcmp(mode,"1")||!strcmp(mode,"unset");bool supported=m==1&&(k==4096||k==2048);
   verify(stats.stream_sd_calls==(size_t)(on&&supported));verify(stats.baseline_sd_calls==(size_t)(m==1&&!(on&&supported)));
   printf("Q4_DEFAULTS pd=%d K=%d N=%d M=%d mode=%s bitwise=1 stream=%zu baseline=%zu shape_fallbacks=%zu\n",pd,k,n,m,mode,stats.stream_sd_calls,stats.baseline_sd_calls,stats.shape_fallbacks);
  }
 }
 printf("Q4_DEFAULTS_PASS\n");
}
