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
int main(int argc,char**argv){using namespace kunlun;require(argc==7);auto *context=ggml_init({1024*1024,nullptr,false});require(context);auto backend=ggml_backend_cpu_init();require(backend);int k=atoi(argv[3]),n=atoi(argv[4]);require((k==2048||k==4096)&&n==32000);int blocks=k/256;size_t original=(size_t)n*blocks*210;
 std::ifstream f(argv[1],std::ios::binary);f.seekg(strtoull(argv[2],nullptr,10));std::vector<unsigned char>raw(original);f.read((char*)raw.data(),raw.size());require((size_t)f.gcount()==raw.size());
 std::vector<float>x(k),cpu(n),exact(n);for(int i=0;i<k;++i)x[i]=sin(i*.123f)*.4f;
 std::vector<unsigned char>qx((size_t)blocks*292);auto host=[&]{quantize_row_q8_K(x.data(),qx.data(),k);
 #pragma omp parallel for num_threads(6)
 for(int r=0;r<n;++r)ggml_vec_dot_q6_K_q8_K(k,&cpu[r],0,raw.data()+(size_t)r*blocks*210,0,qx.data(),0,1);
 };
 host();for(float c:cpu)require(std::isfinite(c));std::vector<double>ct;for(int i=0;i<3;++i){auto s=std::chrono::steady_clock::now();host();ct.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count());}
 #pragma omp parallel for num_threads(6)
 for(int r=0;r<n;++r){double sum=0;for(int b=0;b<blocks;++b){auto w=raw.data()+((size_t)r*blocks+b)*210;float d=half(w[208]|(w[209]<<8));for(int h=0;h<2;++h)for(int l=0;l<32;++l)for(int lane=0;lane<4;++lane){int lo=h*64+l+(lane%2)*32;unsigned v=lane<2?w[lo]&15:w[lo]>>4;int q=(v|(((w[128+h*32+l]>>(lane*2))&3)<<4))-32;float weight=(d*(signed char)w[192+h*8+l/16+lane*2])*q;sum+=double(weight)*x[b*256+h*128+l+lane*32];}}exact[r]=sum;}

 int pd=atoi(argv[6]);select(pd);Q6KWeight weight(pd,k,n);weight.upload(raw.data(),raw.size());Q6KWorkspace ws;Buffer dx(pd,k*4),dy(pd,(n+128)*4);std::vector<float>out(n+128,-12345.f);check(copy(dx.data(),x.data(),k*4,host_to_device));check(copy(dy.data(),out.data(),out.size()*4,host_to_device));auto run=[&]{q6_k_matmul(ws,pd,(float*)dy.data()+64,weight,dx.data());};run();check(copy(out.data(),dy.data(),out.size()*4,device_to_host));for(int i=0;i<64;++i)require(out[i]==-12345&&out[n+64+i]==-12345);
 std::vector<float>xa(k);for(int b=0;b<k;b+=256){float mx=0;for(int j=0;j<256;++j)mx=std::max(mx,std::abs(x[b+j]));float inv=mx>0?127.f/mx:0,s=mx/127.f;for(int j=0;j<256;++j){float v=x[b+j]*inv;xa[b+j]=(int)(v+(v>=0?.5f:-.5f))*s;}}
 double err=0,en=0,ce=0,cn=0,ie=0,in=0,ma=0,cma=0,cmr=0;
 for(int r=0;r<n;++r){double reference=0;for(int bb=0;bb<blocks;++bb){auto*b=raw.data()+((size_t)r*blocks+bb)*210;float d=half(b[208]|b[209]<<8);for(int p=0;p<256;++p){int h=p/128,l=p%32,lane=p%128/32,lo=h*64+l+(lane%2)*32;int q=((lane<2?b[lo]&15:b[lo]>>4)|(((b[128+h*32+l]>>(lane*2))&3)<<4))-32;reference+=double((d*(int8_t)b[192+p/16])*q)*xa[bb*256+p];}}
 float y=out[r+64];require(std::isfinite(y)&&y!=-12345);double dd=y-exact[r];err+=dd*dd;en+=double(exact[r])*exact[r];ma=std::max(ma,std::abs(dd));dd=y-cpu[r];ce+=dd*dd;cn+=double(cpu[r])*cpu[r];cma=std::max(cma,std::abs(dd));if(std::abs(cpu[r])>.01)cmr=std::max(cmr,std::abs(dd/cpu[r]));dd=y-reference;ie+=dd*dd;in+=reference*reference;}
 printf("ACCURACY exact_nmse=%.12g cpu_nmse=%.12g implementation_nmse=%.12g maxabs=%.12g guard=pass cpu_maxabs=%.12g cpu_maxrel_ref_gt_0_01=%.12g\n",err/en,ce/cn,ie/in,ma,cma,cmr);fflush(stdout);require(ie/in<1e-10&&err/en<1e-3);
 std::vector<double>gt;for(int i=0;i<3;++i){auto t=std::chrono::steady_clock::now();run();gt.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}std::sort(gt.begin(),gt.end());std::sort(ct.begin(),ct.end());auto st=weight.statistics();auto wst=ws.statistics();printf("RESULT k=%d n=%d pd=%d CPU_ms=%.6f SDNN_ms=%.6f min_ms=%.6f max_ms=%.6f raw_bytes=%zu resident_bytes=%zu workspace_bytes=%zu pack_ms=%.6f upload_ms=%.6f uploads=%zu\n",k,n,pd,ct[1],gt[1],gt.front(),gt.back(),raw.size(),weight.info().payload_bytes,wst.workspace_bytes,st.pack_ms,st.upload_ms,st.uploads);
 std::fill(x.begin(),x.end(),0);check(copy(dx.data(),x.data(),k*4,host_to_device));run();check(copy(out.data(),dy.data(),out.size()*4,device_to_host));for(int r=0;r<n;++r)require(out[r+64]==0);puts("ZERO_PASS");
 std::vector<unsigned short>hx(k);for(int j=0;j<k;++j){_Float16 h=(_Float16)(sin(j*.17f)*2.f);memcpy(&hx[j],&h,2);x[j]=half(hx[j]);}Buffer dh(pd,k*2);check(copy(dh.data(),hx.data(),k*2,host_to_device));check(copy(dx.data(),x.data(),k*4,host_to_device));run();std::vector<float>gold(n),hf(n);check(copy(gold.data(),(float*)dy.data()+64,n*4,device_to_host));q6_k_matmul(ws,pd,(float*)dy.data()+64,weight,dh.data(),1,InputType::f16);check(copy(hf.data(),(float*)dy.data()+64,n*4,device_to_host));require(memcmp(gold.data(),hf.data(),n*4)==0);puts("F16_F32_BITWISE_PASS");
 Buffer mx(pd,k*8),my(pd,(2*n+128)*4);std::vector<float>xx(2*k,0),yy(2*n+128,-12345.f);std::copy(x.begin(),x.end(),xx.begin());check(copy(mx.data(),xx.data(),xx.size()*4,host_to_device));check(copy(my.data(),yy.data(),yy.size()*4,host_to_device));q6_k_matmul(ws,pd,(float*)my.data()+64,weight,mx.data(),2);check(copy(yy.data(),my.data(),yy.size()*4,device_to_host));require(memcmp(yy.data()+64,gold.data(),n*4)==0);for(int j=0;j<n;++j)require(yy[64+n+j]==0);for(int j=0;j<64;++j)require(yy[j]==-12345&&yy[2*n+64+j]==-12345);require(!q6_k_supported(257,32000)&&!q6_k_supported(k,31));puts("M2_GUARD_AND_DESCRIPTOR_PASS");
}
