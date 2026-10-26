#include <kunlun/k200.h>
#include <xpu/runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <string>
using namespace kunlun;
struct Tensor {
 Tensor4Layout p{};std::vector<unsigned char> raw;Buffer device;
 static size_t span(const Tensor4Layout &p){size_t n=p.type==InputType::f16?2:4;for(int d=0;d<4;++d)n+=(size_t)(p.ne[d]-1)*p.nb[d];return n;}
 Tensor(int pd,Tensor4Layout layout):p(layout),raw(span(p)+256,0xa5),device(pd,raw.size()){require(!*tensor4_reason(p));}
 void *ptr(){return (char*)device.data()+128;}size_t bytes()const{return raw.size()-128;}
 size_t off(int a,int b=0,int c=0,int d=0)const{return 128+(size_t)a*p.nb[0]+(size_t)b*p.nb[1]+(size_t)c*p.nb[2]+(size_t)d*p.nb[3];}
 float get(int a,int b=0,int c=0,int d=0)const{float x;if(p.type==InputType::f32)std::memcpy(&x,raw.data()+off(a,b,c,d),4);else{_Float16 h;std::memcpy(&h,raw.data()+off(a,b,c,d),2);x=h;}return x;}
 void put(float v,int a,int b=0,int c=0,int d=0){if(p.type==InputType::f32)std::memcpy(raw.data()+off(a,b,c,d),&v,4);else{_Float16 h=(_Float16)v;std::memcpy(raw.data()+off(a,b,c,d),&h,2);}}
 void upload(){check(copy(device.data(),raw.data(),raw.size(),host_to_device));check(wait());}
 void download(){check(copy(raw.data(),device.data(),raw.size(),device_to_host));check(wait());}
 int guards()const{
  std::vector<bool>w(raw.size());int bs=p.type==InputType::f16?2:4;
  for(int z=0;z<p.ne[3];++z)for(int h=0;h<p.ne[2];++h)for(int r=0;r<p.ne[1];++r)for(int c=0;c<p.ne[0];++c)for(int j=0;j<bs;++j)w[off(c,r,h,z)+j]=true;
  int errors=0;for(size_t i=0;i<raw.size();++i)if(!w[i] && raw[i]!=0xa5)++errors;return errors;
 }
};
static Tensor4Layout dense(int x,int y,int h,int b,InputType type=InputType::f32,int gap=0){int bs=type==InputType::f16?2:4;return {{x,y,h,b},{bs,x*bs+gap,(x*bs+gap)*y,(x*bs+gap)*y*h},type};}
static float sample(int i){return ((i*37%1009)-504)/503.f;}
struct Error {double sq=0,en=0,max=0;int bad=0;void add(float a,double b){if(!std::isfinite(a)||!std::isfinite(b)){++bad;return;}double d=a-b;sq+=d*d;en+=b*b;max=std::max(max,std::abs(d));}double nmse()const{return sq/std::max(en,1e-30);}};
int main(int argc,char **argv){
 int pd=argc>1?atoi(argv[1]):0,d=64,s=512,hk=4,h=32,f=d*hk,cap=s+64;auto qp=dense(d,1,h,1),yp=dense(d,h,1,1),kp=dense(d,s,hk,1,InputType::f16);kp.nb[1]=f*2;kp.nb[2]=d*2;kp.nb[3]=f*s*2;
 Tensor q(pd,qp),a(pd,yp),b(pd,yp);Buffer storage(pd,(size_t)cap*f*2);KVMaxCache owner(pd,storage.data(),storage.size(),f,cap,"alias-KV");
 std::vector<_Float16>data((size_t)cap*f);for(size_t j=0;j<data.size();++j)data[j]=(_Float16)sinf(j*.17f);data[0]=32;data[(s+1)*f]=64;check(copy(storage.data(),data.data(),storage.size(),host_to_device));check(wait());for(int j=0;j<h;++j)for(int x=0;x<d;++x)q.put(cosf(x*.07f+j*.3f),x,0,j);q.upload();Flash4Layout p{};p.q=qp;p.k=p.v=kp;p.y=yp;p.scale=.125f;Kernel kernel;
 for(auto off:std::vector<std::pair<int,int>>{{0,32},{32,32},{0,0},{32,0},{0,0}}){
  auto run=[&](const char *mode,Tensor &out){setenv("KUNLUN_FLASH_KV_MAX",mode,1);size_t ko=(size_t)off.first*f*2,vo=(size_t)off.second*f*2;require(flash_attention(kernel,pd,out.ptr(),out.bytes(),q.ptr(),q.bytes(),(char*)storage.data()+ko,storage.size()-ko,(char*)storage.data()+vo,storage.size()-vo,nullptr,0,p));};
  run("verify",a);run("legacy",b);a.download();b.download();for(int j=0;j<h;++j)for(int x=0;x<d;++x)require(!memcmp(a.raw.data()+a.off(x,j),b.raw.data()+b.off(x,j),4));require(kv_max_statistics(pd).hits==0);printf("KV_ALIAS pd=%d kstart=%d vstart=%d bits=0 fallback=1\n",pd,off.first,off.second);
 }
 Kernel writer;Buffer indices(pd,256);int64_t target=0;check(copy(indices.data(),&target,8,host_to_device));check(wait());SetRowsLayout w{};w.width=f;w.rows=w.dim2=w.dim3=1;w.source_stride[0]=2;w.source_stride[1]=w.source_stride[2]=w.source_stride[3]=f*2;w.destination_stride[0]=f*2;w.destination_stride[1]=w.destination_stride[2]=cap*f*2;w.index_stride[0]=w.index_stride[1]=w.index_stride[2]=8;w.index_extent[0]=w.index_extent[1]=1;w.source_bytes=2;w.index_bytes=8;w.destination_rows=cap;
 require(!kv_max_set_rows(writer,pd,storage.data(),storage.size()-2,q.ptr(),q.bytes(),indices.data(),indices.size(),w,&target,1));require(!kv_max_set_rows(writer,pd,storage.data(),storage.size(),storage.data(),storage.size(),indices.data(),indices.size(),w,&target,1));printf("KV_WRITER_GUARD pd=%d short_span=1 source_alias=1 rejected_before_launch=1\n",pd);
 std::vector<_Float16>after(data.size());check(copy(after.data(),storage.data(),storage.size(),device_to_host));check(wait());require(!memcmp(after.data(),data.data(),storage.size()));
 // PP checked conversion must flag Inf/NaN and finite F32 values that round to
 // F16 Inf. After a finite overwrite poison must still force SDK findmax.
 Buffer vd(pd,storage.size()),src(pd,f*2*4);KVMaxCache vowner(pd,vd.data(),vd.size(),f,cap,"nonfinite-V");check(copy(vd.data(),data.data(),vd.size(),host_to_device));check(wait());int64_t ids[2]={0,1};check(copy(indices.data(),ids,16,host_to_device));check(wait());
 std::vector<_Float16>zeros(data.size());std::vector<float>values(f*2,.5f);std::vector<unsigned short>halfvalues(f*2,0x3800);w.rows=2;
 for(int trial=0;trial<4;++trial){
  check(copy(storage.data(),zeros.data(),storage.size(),host_to_device));check(wait());kv_max_zero(pd,storage.data(),storage.size());setenv("KUNLUN_FLASH_KV_MAX","verify",1);w.source_bytes=trial==3?2:4;w.source_stride[0]=w.source_bytes;w.source_stride[1]=f*w.source_bytes;
  if(trial==3){halfvalues[0]=0x7e00;check(copy(src.data(),halfvalues.data(),f*4,host_to_device));}else{values[0]=trial==0?INFINITY:trial==1?65520.f:NAN;check(copy(src.data(),values.data(),f*8,host_to_device));}check(wait());require(kv_max_set_rows(writer,pd,storage.data(),storage.size(),src.data(),src.size(),indices.data(),indices.size(),w,ids,2));unsigned short bits=0;check(copy(&bits,storage.data(),2,device_to_host));check(wait());require((bits&0x7fff)>=0x7c00);
  if(trial==3){halfvalues[0]=0x3800;check(copy(src.data(),halfvalues.data(),f*4,host_to_device));}else{values[0]=.5f;check(copy(src.data(),values.data(),f*8,host_to_device));}check(wait());require(kv_max_set_rows(writer,pd,storage.data(),storage.size(),src.data(),src.size(),indices.data(),indices.size(),w,ids,2));
  size_t before=kv_max_statistics(pd).fallback_poison;require(flash_attention(kernel,pd,a.ptr(),a.bytes(),q.ptr(),q.bytes(),storage.data(),storage.size(),vd.data(),vd.size(),nullptr,0,p));require(kv_max_statistics(pd).fallback_poison>before);setenv("KUNLUN_FLASH_KV_MAX","legacy",1);require(flash_attention(kernel,pd,b.ptr(),b.bytes(),q.ptr(),q.bytes(),storage.data(),storage.size(),vd.data(),vd.size(),nullptr,0,p));a.download();b.download();for(int j=0;j<h;++j)for(int x=0;x<d;++x)require(!memcmp(a.raw.data()+a.off(x,j),b.raw.data()+b.off(x,j),4));printf("KV_WRITER_NONFINITE pd=%d trial=%d flagged=1 finite_overwrite_stays_poison=1 bits=0\n",pd,trial);
 }
 kv_max_log_statistics();
}
