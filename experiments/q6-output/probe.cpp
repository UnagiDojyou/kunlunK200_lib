#include <kunlun/k200.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include <xpu/runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <omp.h>
extern "C" void quantize_row_q8_K(const float*,void*,int64_t);
extern "C" void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
static float half(unsigned short u){_Float16 h;memcpy(&h,&u,2);return h;}
int main(int argc,char**argv){using namespace kunlun;require(argc==6);auto *context=ggml_init({1024*1024,nullptr,false});require(context);auto backend=ggml_backend_cpu_init();require(backend);int k=atoi(argv[3]),n=atoi(argv[4]);require((k==2048||k==4096)&&n==32000);int blocks=k/256;size_t original=(size_t)n*blocks*210;
 std::ifstream f(argv[1],std::ios::binary);f.seekg(strtoull(argv[2],nullptr,10));std::vector<unsigned char>raw(original);f.read((char*)raw.data(),raw.size());require((size_t)f.gcount()==raw.size());
 std::vector<unsigned char>packed((size_t)n*blocks*256,0);for(size_t i=0;i<(size_t)n*blocks;++i)memcpy(packed.data()+i*256,raw.data()+i*210,210);
 std::vector<float>x(k),cpu(n),exact(n);for(int i=0;i<k;++i)x[i]=sin(i*.123f)*.4f;
 std::vector<unsigned char>qx((size_t)blocks*292);auto host=[&]{quantize_row_q8_K(x.data(),qx.data(),k);
 #pragma omp parallel for num_threads(6)
 for(int r=0;r<n;++r)ggml_vec_dot_q6_K_q8_K(k,&cpu[r],0,raw.data()+(size_t)r*blocks*210,0,qx.data(),0,1);
 };
 host();for(float c:cpu)require(std::isfinite(c));std::vector<double>ct;for(int i=0;i<3;++i){auto s=std::chrono::steady_clock::now();host();ct.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count());}
 #pragma omp parallel for num_threads(6)
 for(int r=0;r<n;++r){double sum=0;for(int b=0;b<blocks;++b){auto w=raw.data()+((size_t)r*blocks+b)*210;float d=half(w[208]|(w[209]<<8));for(int h=0;h<2;++h)for(int l=0;l<32;++l)for(int lane=0;lane<4;++lane){int lo=h*64+l+(lane%2)*32;unsigned v=lane<2?w[lo]&15:w[lo]>>4;int q=(v|(((w[128+h*32+l]>>(lane*2))&3)<<4))-32;float weight=(d*(signed char)w[192+h*8+l/16+lane*2])*q;sum+=double(weight)*x[b*256+h*128+l+lane*32];}}exact[r]=sum;}
 select(0);Buffer w(0,packed.size()),dx(0,x.size()*4),dy(0,(n+128)*4);
 auto upload=std::chrono::steady_clock::now();check(copy(w.data(),packed.data(),packed.size(),host_to_device));check(copy(dx.data(),x.data(),x.size()*4,host_to_device));check(wait());double upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-upload).count();
 std::string dir=argv[5];std::ifstream bin(dir+"/kunlun-q6-probe.bin",std::ios::binary),entry(dir+"/kunlun-q6-probe.entry");std::vector<char>image((std::istreambuf_iterator<char>(bin)),{});std::string symbol;getline(entry,symbol);require(!image.empty()&&image.size()<16384&&!symbol.empty());image.resize((image.size()+255)/256*256+256,0);Buffer code(0,image.size());check(copy(code.data(),image.data(),image.size(),host_to_device));check(wait());XPUFunc fn{};check(xpu_create_cl_func(&fn,(uint64_t)(uintptr_t)code.data(),image.size(),0,0,0,symbol.c_str(),true));
 auto run=[&]{std::lock_guard<std::recursive_mutex>lock(runtime_mutex());float*y=(float*)dy.data()+64;const void*wp=w.data(),*xp=dx.data();size_t off=0;for(const void*p:{(const void*)&y,(const void*)&wp,(const void*)&xp}){check(xpu_launch_argument_set(p,8,off));off+=8;}for(const int*p:{&k,&n}){check(xpu_launch_argument_set(p,4,off));off+=4;}check(xpu_launch_config(4,16));check(xpu_launch_async(fn));check(wait());};
 std::vector<float>sentinel(n+128,-12345.f),out(n+128);check(copy(dy.data(),sentinel.data(),sentinel.size()*4,host_to_device));run();check(copy(out.data(),dy.data(),out.size()*4,device_to_host));for(int i=0;i<64;++i)require(out[i]==sentinel[i]&&out[n+64+i]==sentinel[n+64+i]);
 double sq=0,en=0,ma=0,mr=0,csq=0,cen=0;for(int i=0;i<n;++i){float y=out[i+64];require(std::isfinite(y)&&y!=-12345.f);double d=y-exact[i];sq+=d*d;en+=double(exact[i])*exact[i];ma=std::max(ma,std::abs(d));if(std::abs(exact[i])>.01)mr=std::max(mr,std::abs(d/exact[i]));double e=y-cpu[i];csq+=e*e;cen+=double(cpu[i])*cpu[i];}require(sq/en<1e-9 && cen>0);
 std::vector<double>gt;for(int i=0;i<3;++i){auto s=std::chrono::steady_clock::now();run();gt.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count());}std::sort(gt.begin(),gt.end());std::sort(ct.begin(),ct.end());
 printf("RESULT k=%d n=%d CPU_ms=%.6f XPU_ms=%.6f XPU_min=%.6f XPU_max=%.6f exact_nmse=%.12g max_abs=%.12g max_rel_ref_gt_0_01=%.12g ggml_q8_nmse=%.12g raw_bytes=%zu resident_bytes=%zu upload_ms=%.6f guard=pass\n",k,n,ct[1],gt[1],gt.front(),gt.back(),sq/en,ma,mr,csq/cen,raw.size(),packed.size(),upload_ms);
 // Zero input confirms every output is written, with the same guards.
 std::fill(x.begin(),x.end(),0);check(copy(dx.data(),x.data(),x.size()*4,host_to_device));run();check(copy(out.data(),dy.data(),out.size()*4,device_to_host));for(int i=0;i<n;++i)require(out[i+64]==0);printf("ZERO_PASS\n");
}
