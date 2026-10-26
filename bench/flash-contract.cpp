#include <kunlun/k200.h>
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
int main(){
 Flash4Layout p{};p.q=dense(64,33,8,2);p.k=dense(64,288,2,1,InputType::f16);p.v=p.k;p.y=dense(64,8,33,2);p.mask=dense(288,33,2,1);p.has_mask=true;p.scale=.125f;
 require(!*flash_attention_reason(p));int rejected=0;
 auto test=[&](Flash4Layout bad){require(*flash_attention_reason(bad));++rejected;};
 auto z=p;z.q.ne[0]=96;test(z);z=p;z.k.ne[1]=257;test(z);z=p;z.q.ne[1]=257;test(z);z=p;z.q.ne[2]=7;test(z);z=p;z.y.type=InputType::f16;test(z);z=p;z.k.type=InputType::f32;test(z);z=p;z.scale=NAN;test(z);z=p;z.max_bias=-1;test(z);z=p;z.mask.ne[2]=3;test(z);z=p;z.q.nb[2]=4;test(z);
 Kernel no_submit;require(!flash_attention(no_submit,2,nullptr,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0,p));
 Buffer b(0,1024*1024);void *base=b.data();require(!flash_attention(no_submit,0,base,b.size(),(char*)base+2,b.size()-2,base,b.size(),base,b.size(),base,b.size(),p));
 printf("FLASH_CONTRACT rejected_shapes=%d invalid_pointer_and_device=pass\n",rejected);
 int checks=0;
 for(int pd=0;pd<2;++pd)for(int width:{37,257})for(int sb:{2,4})for(int ib:{4,8})for(int duplicate:{0,1}){
  auto sl=dense(width,5,1,1,sb==2?InputType::f16:InputType::f32,12),dl=dense(width,8,1,1,InputType::f16,14);
  Tensor src(pd,sl),dst(pd,dl);for(int row=0;row<5;++row)for(int x=0;x<width;++x)src.put((row*3+x%17)*.03125f,x,row);src.upload();dst.upload();auto original=src.raw,expected=dst.raw;
  int target[5]={2,3,4,5,6};if(duplicate){target[0]=1;target[1]=2;target[2]=2;target[3]=0;target[4]=7;}
  std::vector<unsigned char>indices(5*ib);for(int row=0;row<5;++row){int64_t idx=target[row];memcpy(indices.data()+row*ib,&idx,ib);for(int x=0;x<width;++x){_Float16 value=(_Float16)src.get(x,row);memcpy(expected.data()+dst.off(x,target[row]),&value,2);}}
  Buffer ids(pd,indices.size());check(copy(ids.data(),indices.data(),indices.size(),host_to_device));check(wait());
  SetRowsLayout kv{};kv.width=width;kv.rows=5;kv.dim2=kv.dim3=1;for(int d=0;d<4;++d)kv.source_stride[d]=sl.nb[d];for(int d=0;d<3;++d)kv.destination_stride[d]=dl.nb[d+1];kv.index_stride[0]=ib;kv.index_stride[1]=5*ib;kv.index_stride[2]=5*ib;kv.index_extent[0]=kv.index_extent[1]=1;kv.source_bytes=sb;kv.index_bytes=ib;kv.destination_rows=8;
  Kernel kernel;setenv("KUNLUN_FLASH_KV","1",1);require(set_rows_f16(kernel,pd,dst.ptr(),dst.bytes(),src.ptr(),src.bytes(),ids.data(),ids.size(),kv));dst.download();src.download();require(dst.raw==expected&&src.raw==original);
  printf("FLASH_KV_CONTRACT pd=%d width=%d source_bytes=%d index_bytes=%d duplicate=%d pass=1\n",pd,width,sb,ib,duplicate);++checks;
 }
 printf("FLASH_KV_CONTRACT checks=%d pass=1\n",checks);return 0;
}
