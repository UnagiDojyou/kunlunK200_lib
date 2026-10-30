#include <kunlun/k200.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include <xpu/runtime.h>
#include <vector>
#include <fstream>
#include <iterator>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <algorithm>
extern "C" void quantize_row_q8_K(const float*,void*,int64_t);
static float half(unsigned short u){_Float16 h;memcpy(&h,&u,2);return h;}
int main(int argc,char**argv){using namespace kunlun;require(argc==8);auto*ctx=ggml_init({1024*1024,nullptr,false});require(ctx);require(ggml_backend_cpu_init());int fullk=atoi(argv[3]),k=atoi(argv[4]),n=atoi(argv[5]);require(k%256==0&&k<=fullk&&k<=4096&&n%32==0);int groups=k/16,nt=n/32;
 std::ifstream f(argv[1],std::ios::binary);size_t start=strtoull(argv[2],nullptr,10);std::vector<unsigned char>raw((size_t)n*k/256*210);for(int r=0;r<n;++r){f.seekg(start+(size_t)r*fullk/256*210);f.read((char*)raw.data()+(size_t)r*k/256*210,k/256*210);require(f.good());}
 std::vector<int8_t>w((size_t)groups*nt*1024,0),a(groups*1024,0);std::vector<float>sc(groups*n),as(groups),x(k);for(int i=0;i<k;++i)x[i]=sin(i*.123f)*.4f;std::vector<unsigned char>qx(k/256*292);quantize_row_q8_K(x.data(),qx.data(),k);
 for(int g=0;g<groups;++g){memcpy(&as[g],qx.data()+g/16*292,4);for(int rr=0;rr<32;++rr)memcpy(a.data()+g*1024+rr*32,qx.data()+g/16*292+4+g%16*16,16);for(int r=0;r<n;++r){auto b=raw.data()+((size_t)r*k/256+g/16)*210;sc[g*n+r]=half(b[208]|b[209]<<8)*(int8_t)b[192+g%16];for(int j=0;j<16;++j){int p=g%16*16+j,h=p/128,rem=p%128,l=rem%32,lane=rem/32,lo=h*64+l+(lane%2)*32;int q=((lane<2?b[lo]&15:b[lo]>>4)|(((b[128+h*32+l]>>(lane*2))&3)<<4))-32;w[((size_t)g*nt+r/32)*1024+j*32+r%32]=q;}}}
 select(0);Buffer dw(0,w.size()),da(0,a.size()),ds(0,sc.size()*4),das(0,as.size()*4),dp(0,((size_t)groups*n+128)*4);check(copy(dw.data(),w.data(),w.size(),host_to_device));check(copy(da.data(),a.data(),a.size(),host_to_device));check(copy(ds.data(),sc.data(),sc.size()*4,host_to_device));check(copy(das.data(),as.data(),as.size()*4,host_to_device));check(wait());std::vector<float>out((size_t)groups*n+128,-12345);check(copy(dp.data(),out.data(),out.size()*4,host_to_device));
 std::string dir=argv[6];std::ifstream bin(dir+"/kunlun-q6-sdnn-probe.bin",std::ios::binary),ent(dir+"/kunlun-q6-sdnn-probe.entry");std::vector<char>im((std::istreambuf_iterator<char>(bin)),{});std::string sym;getline(ent,sym);require(im.size()>0&&im.size()<16384);im.resize((im.size()+255)/256*256+256,0);Buffer code(0,im.size());check(copy(code.data(),im.data(),im.size(),host_to_device));check(wait());XPUFunc fn{};check(xpu_create_sd_func(&fn,(uint64_t)(uintptr_t)code.data(),im.size(),0,0,0,sym.c_str(),true));
 auto run=[&]{std::lock_guard<std::recursive_mutex>lock(runtime_mutex());float*p=(float*)dp.data()+64;const void*wp=dw.data(),*ap=da.data(),*sp=ds.data(),*asp=das.data();size_t off=0;for(const void*v:{(void*)&p,(void*)&wp,(void*)&ap}){check(xpu_launch_argument_set(v,8,off));off+=8;}for(const int*v:{&k,&n}){check(xpu_launch_argument_set(v,4,off));off+=4;}for(const void*v:{(void*)&sp,(void*)&asp}){check(xpu_launch_argument_set(v,8,off));off+=8;}check(xpu_launch_config(4,8));check(xpu_launch_async(fn));check(wait());};run();check(copy(out.data(),dp.data(),out.size()*4,device_to_host));for(int i=0;i<64;++i)require(out[i]==-12345&&out[64+(size_t)groups*n+i]==-12345);
 double err=0,en=0,ma=0;int bad=0;for(int g=0;g<groups;++g)for(int r=0;r<n;++r){int sum=0;for(int j=0;j<16;++j)sum+=(int)w[((size_t)g*nt+r/32)*1024+j*32+r%32]*(int)a[g*1024+j];float ref=(float(sum)*as[g])*sc[g*n+r],y=out[64+g*n+r];require(std::isfinite(y));double d=y-ref;err+=d*d;en+=double(ref)*ref;ma=std::max(ma,std::abs(d));if(std::abs(d)>1e-5&&bad++<8)printf("DIFF g=%d r=%d sum=%d ref=%g got=%g\n",g,r,sum,ref,y);}
 printf("PARTIAL k=%d n=%d nmse=%.12g maxabs=%.12g bad=%d guard=pass\n",k,n,err/en,ma,bad);fflush(stdout);if(err/en>1e-10)return 2;
 std::vector<double>times;for(int i=0;i<atoi(argv[7]);++i){auto t=std::chrono::steady_clock::now();run();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}std::sort(times.begin(),times.end());if(!times.empty())printf("TIME median_ms=%.6f min_ms=%.6f max_ms=%.6f wbytes=%zu scale_bytes=%zu partial_bytes=%zu\n",times[times.size()/2],times.front(),times.back(),w.size(),sc.size()*4,(size_t)groups*n*4);
}
