#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
static void dequantize_row_q4_0(const void *raw,float *y,int64_t k) {
 const auto *p=(const unsigned char*)raw;
 for(int b=0;b<k/32;++b){unsigned short bits=p[b*18]|p[b*18+1]<<8;_Float16 h;std::memcpy(&h,&bits,2);float scale=h;for(int j=0;j<16;++j){int q=p[b*18+2+j];y[b*32+j]=float((q&15)-8)*scale;y[b*32+j+16]=float((q>>4)-8)*scale;}}
}
static double ms(std::chrono::steady_clock::time_point start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
int main(int argc,char**argv){
 using namespace kunlun;require(argc==7);
 int k=atoi(argv[3]),n=atoi(argv[4]),pd=atoi(argv[5]),timing=atoi(argv[6]);size_t row_bytes=(size_t)k/32*18;
 std::vector<unsigned char>raw(row_bytes*n);std::ifstream f(argv[1],std::ios::binary);f.seekg(strtoull(argv[2],nullptr,10));f.read((char*)raw.data(),raw.size());require((size_t)f.gcount()==raw.size());
 CommonQ4Weight w(pd,k,n);w.upload(raw.data(),raw.size());Kernel kernel;
 Buffer ids(pd,512*4+512),output(pd,(512*k+128)*4);std::vector<int>indices(512);std::vector<float>reference(512*k),values(512*k+128,-12345.f);select(pd);
 for(int m:{1,3,128,512}){
  for(int i=0;i<m;++i)indices[i]=(i%7==0)?n-1:(i%5==0)?0:(i*1237+13)%n;
  auto cpu=[&]{
   for(int i=0;i<m;++i)dequantize_row_q4_0(raw.data()+indices[i]*row_bytes,reference.data()+(size_t)i*k,k);
  };cpu();
  auto *di=(int*)ids.data()+64;auto *dy=(float*)output.data()+64;
  check(copy(di,indices.data(),m*4,host_to_device));check(copy(output.data(),values.data(),((size_t)m*k+128)*4,host_to_device));
  auto gpu=[&]{require(common_q4_get_rows(kernel,pd,dy,(size_t)m*k*4,w,di,m*4,m));};gpu();check(copy(values.data(),output.data(),((size_t)m*k+128)*4,device_to_host));
  for(int i=0;i<64;++i)require(values[i]==-12345.f&&values[(size_t)m*k+64+i]==-12345.f);
  for(size_t i=0;i<(size_t)m*k;++i)require(std::isfinite(values[64+i]));
  if(memcmp(reference.data(),values.data()+64,(size_t)m*k*4)){
   for(size_t i=0;i<(size_t)m*k;++i)if(memcmp(&reference[i],&values[64+i],4)){printf("DIFF m=%d i=%zu ref=%g got=%g\n",m,i,reference[i],values[64+i]);break;}return 2;
  }
  std::vector<double>ct,gt,oldcost,newcost;
  if(timing&&m!=3)for(int r=0;r<3;++r){auto t=std::chrono::steady_clock::now();cpu();ct.push_back(ms(t));t=std::chrono::steady_clock::now();gpu();gt.push_back(ms(t));t=std::chrono::steady_clock::now();cpu();check(copy(dy,reference.data(),(size_t)m*k*4,host_to_device));check(wait());oldcost.push_back(ms(t));t=std::chrono::steady_clock::now();check(copy(di,indices.data(),m*4,host_to_device));gpu();newcost.push_back(ms(t));}
  auto median=[](std::vector<double>x){if(x.empty())return 0.;std::sort(x.begin(),x.end());return x[x.size()/2];};
  printf("GET_ROWS m=%d k=%d n=%d pd=%d bitwise=1 finite=1 guard=1 cpu_serial_ms=%.6f gpu_ms=%.6f cpu_plus_h2d_ms=%.6f ids_plus_gpu_ms=%.6f\n",m,k,n,pd,median(ct),median(gt),median(oldcost),median(newcost));fflush(stdout);
  std::fill(values.begin(),values.end(),-12345.f);
 }
 auto*di=(int*)ids.data()+64;auto*dy=(float*)output.data()+64;int invalid=-1;check(copy(di,&invalid,4,host_to_device));check(copy(output.data(),values.data(),(k+128)*4,host_to_device));require(!common_q4_get_rows(kernel,pd,dy,k*4,w,di,4,1));check(copy(values.data(),output.data(),(k+128)*4,device_to_host));for(int i=0;i<k+128;++i)require(values[i]==-12345.f);
 invalid=n;check(copy(di,&invalid,4,host_to_device));require(!common_q4_get_rows(kernel,pd,dy,k*4,w,di,4,1));require(!common_q4_get_rows(kernel,pd,dy,k*4-1,w,di,4,1));require(!common_q4_get_rows(kernel,pd,dy,(size_t)512*k*4,w,di,512*4,513));
 auto a=w.info();auto s=w.statistics();printf("GET_ROWS_UNIT_PASS invalid_ids_rejected=1 payload=%zu uploads=%zu pack_ms=%.3f upload_ms=%.3f\n",a.payload_bytes,s.uploads,s.pack_ms,s.upload_ms);
}
