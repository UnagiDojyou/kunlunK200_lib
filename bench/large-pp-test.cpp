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

static void flash_test(int pd,int d,int m,int s,int h,int hk,bool padded,bool mask_half,int batch=1) {
 auto qp=dense(d,m,h,batch),kp=dense(d,s,hk,1,InputType::f16);
 if(!padded){qp.nb[1]=d*h*4;qp.nb[2]=d*4;qp.nb[3]=d*h*m*4;kp.nb[1]=d*hk*2;kp.nb[2]=d*2;kp.nb[3]=d*hk*s*2;}
 else{qp.nb[1]+=32;qp.nb[2]=qp.nb[1]*m+32;qp.nb[3]=qp.nb[2]*h+32;}
 Tensor q(pd,qp),k(pd,kp),v(pd,kp),out(pd,dense(d,h,m,batch)),manual(pd,out.p);
 Tensor mask(pd,dense(s,m,padded?2:1,batch,mask_half?InputType::f16:InputType::f32));
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int t=0;t<m;++t)for(int x=0;x<d;++x)q.put(sample(x+17*t+31*j+3*b),x,t,j,b);
 for(int j=0;j<hk;++j)for(int t=0;t<s;++t)for(int x=0;x<d;++x){k.put(sample(3*x+19*t+23*j),x,t,j);v.put(sample(13*x+7*t+11*j),x,t,j);}
 for(int b=0;b<batch;++b)for(int j=0;j<mask.p.ne[2];++j)for(int t=0;t<m;++t)for(int z=0;z<s;++z)mask.put(z>s-m+t||(padded&&t==0)?-INFINITY:-.002f*(z%7)*(j+1),z,t,j,b);
 auto qr=q.raw,kr=k.raw,vr=v.raw,mr=mask.raw;q.upload();k.upload();v.upload();mask.upload();out.upload();manual.upload();
 Flash4Layout p{};p.q=q.p;p.k=k.p;p.v=v.p;p.y=out.p;p.mask=mask.p;p.has_mask=true;p.scale=1.f/sqrtf(d);p.max_bias=padded?8:0;
 setenv("KUNLUN_LARGE_PP","0",1);if(m>256)require(*flash_attention_reason(p));
 setenv("KUNLUN_LARGE_PP","1",1);require(!*flash_attention_reason(p));auto bad=p;bad.q.ne[0]=96;require(*flash_attention_reason(bad));bad=p;bad.scale=NAN;require(*flash_attention_reason(bad));
 Kernel full,reference;require(flash_attention(full,pd,out.ptr(),out.bytes(),q.ptr(),q.bytes(),k.ptr(),k.bytes(),v.ptr(),v.bytes(),mask.ptr(),mask.bytes(),p));
 setenv("KUNLUN_LARGE_PP","0",1);
 for(int first=0;first<m;first+=128){int n=std::min(128,m-first);auto part=p;part.q.ne[1]=n;part.y.ne[2]=n;part.mask.ne[1]=n;
  size_t qo=first*(size_t)qp.nb[1],yo=first*(size_t)p.y.nb[2],mo=first*(size_t)p.mask.nb[1];
  require(flash_attention(reference,pd,(char*)manual.ptr()+yo,manual.bytes()-yo,(char*)q.ptr()+qo,q.bytes()-qo,k.ptr(),k.bytes(),v.ptr(),v.bytes(),(char*)mask.ptr()+mo,mask.bytes()-mo,part));
 }
 out.download();manual.download();require(out.raw==manual.raw && !out.guards());
 Error e;
 for(int b=0;b<batch;++b)for(int head=0;head<h;++head)for(int t:{0,127,128,255,m-1})if(t<m){
  std::vector<double>w(s);double maximum=-INFINITY,sum=0;float slope=1;
  if(p.max_bias){int nh=1;while(nh*2<=h)nh*=2;float a=powf(2,-p.max_bias/nh),c=powf(2,-p.max_bias/(2*nh));slope=head<nh?powf(a,head+1):powf(c,2*(head-nh)+1);}
  for(int z=0;z<s;++z){double dot=0;for(int x=0;x<d;++x)dot+=(float)(_Float16)q.get(x,t,head,b)*(double)k.get(x,z,head/(h/hk));w[z]=dot*p.scale+mask.get(z,t,head%mask.p.ne[2],b)*slope;maximum=std::max(maximum,w[z]);}
  for(double &x:w){x=x==-INFINITY?0:std::exp(x-maximum);sum+=x;}
  for(int x=0;x<d;++x){double expected=0;if(sum)for(int z=0;z<s;++z)expected+=w[z]/sum*v.get(x,z,head/(h/hk));e.add(out.get(x,head,t,b),expected);}
 }
 require(!e.bad && e.nmse()<2e-4);
 Tensor flat(pd,dense(d,m,h,batch));flat.upload();Kernel copy_kernel;require(attention_contiguous(copy_kernel,pd,flat.ptr(),flat.bytes(),q.ptr(),q.bytes(),q.p));flat.download();
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int t=0;t<m;++t)for(int x=0;x<d;++x)require(flat.get(x,t,j,b)==q.get(x,t,j,b));
 q.download();k.download();v.download();mask.download();require(q.raw==qr&&k.raw==kr&&v.raw==vr&&mask.raw==mr&&!flat.guards());
 printf("LARGE_FLASH pd=%d D=%d M=%d S=%d H=%d HK=%d B=%d padded=%d mask_half=%d manual_bitwise=1 cpu_nmse=%.12g max_abs=%.12g cont_bitwise=1 guards=pass\n",pd,d,m,s,h,hk,batch,padded,mask_half,e.nmse(),e.max);
}
static void rope_test(int pd,int d,int h,int m,InputType type,int mode) {
 auto layout=dense(d,h,m,1,type,16);Tensor x(pd,layout),y(pd,layout),reference(pd,layout);
 for(int t=0;t<m;++t)for(int j=0;j<h;++j)for(int a=0;a<d;++a)x.put(sample(a+j*7+t*11),a,j,t);
 x.upload();y.upload();reference.upload();std::vector<int32_t>positions(m);for(int t=0;t<m;++t)positions[t]=t+319;Buffer ids(pd,m*4);check(copy(ids.data(),positions.data(),m*4,host_to_device));check(wait());
 Rope4Layout p{layout,layout,d,mode,0,10000,1,1};Kernel full,old;
 require(attention_rope(full,pd,y.ptr(),y.bytes(),x.ptr(),x.bytes(),(int32_t*)ids.data(),ids.size(),p));
 for(int first=0;first<m;first+=128){auto part=p;part.x.ne[2]=part.y.ne[2]=std::min(128,m-first);size_t offset=first*(size_t)layout.nb[2];
  require(attention_rope(old,pd,(char*)reference.ptr()+offset,reference.bytes()-offset,(char*)x.ptr()+offset,x.bytes()-offset,(int32_t*)ids.data()+first,ids.size()-first*4,part));
 }
 y.download();reference.download();require(y.raw==reference.raw&&!y.guards());
 require(attention_rope(full,pd,x.ptr(),x.bytes(),x.ptr(),x.bytes(),(int32_t*)ids.data(),ids.size(),p));x.download();require(x.raw==y.raw);
 p.x.nb[0]*=2;require(*attention_rope_reason(p));printf("LARGE_ROPE pd=%d M=%d type=%d mode=%d sliced_bitwise=1 inplace_bitwise=1 guards=pass\n",pd,m,(int)type,mode);
}
static void kv_test(int pd,int source_bytes,int index_bytes) {
 const int features=256,rows=1024,count=513;auto view=dense(64,rows,4,1,InputType::f16);view.nb[1]=features*2;view.nb[2]=64*2;view.nb[3]=features*rows*2;
 Tensor key(pd,view),value(pd,view),source(pd,dense(features,count,1,1,source_bytes==2?InputType::f16:InputType::f32));
 for(int t=0;t<rows;++t)for(int h=0;h<4;++h)for(int x=0;x<64;++x){key.put(0,x,t,h);value.put(0,x,t,h);}
 for(int t=0;t<count;++t)for(int x=0;x<features;++x)source.put(sample(t*7+x),x,t);
 key.upload();value.upload();source.upload();size_t bytes=(size_t)features*rows*2;
 KVMaxCache kc(pd,key.ptr(),bytes,features,rows,"large-K"),vc(pd,value.ptr(),bytes,features,rows,"large-V");kv_max_zero(pd,key.ptr(),bytes);kv_max_zero(pd,value.ptr(),bytes);
 std::vector<int64_t>index(count),targets(count);for(int t=0;t<count;++t){index[t]=32+(t*7)%count;targets[t]=32+t;}std::vector<unsigned char> raw_ids(count*index_bytes);for(int t=0;t<count;++t)memcpy(raw_ids.data()+t*index_bytes,&index[t],index_bytes);Buffer ids(pd,raw_ids.size());check(copy(ids.data(),raw_ids.data(),raw_ids.size(),host_to_device));check(wait());
 SetRowsLayout p{};p.width=features;p.rows=count;p.dim2=p.dim3=1;std::copy(source.p.nb,source.p.nb+4,p.source_stride);p.destination_stride[0]=features*2;p.destination_stride[1]=p.destination_stride[2]=features*rows*2;p.index_stride[0]=index_bytes;p.index_stride[1]=p.index_stride[2]=count*index_bytes;p.index_extent[0]=p.index_extent[1]=1;p.source_bytes=source_bytes;p.index_bytes=index_bytes;p.destination_rows=rows;
 Kernel writer;setenv("KUNLUN_FLASH_KV_MAX","verify",1);setenv("KUNLUN_FLASH_KV_MAX_PREFILL","fused",1);setenv("KUNLUN_LARGE_PP","0",1);require(!kv_max_set_rows(writer,pd,key.ptr(),bytes,source.ptr(),source.bytes(),ids.data(),ids.size(),p,targets.data(),count));setenv("KUNLUN_LARGE_PP","1",1);
 auto write=[&](Tensor &dst){require(kv_max_set_rows(writer,pd,dst.ptr(),bytes,source.ptr(),source.bytes(),ids.data(),ids.size(),p,targets.data(),count));};write(key);write(value);key.download();value.download();
 for(int t=0;t<count;++t)for(int x=0;x<features;++x)require(key.get(x%64,index[t],x/64)==(float)(_Float16)source.get(x,t));require(!key.guards()&&!value.guards());
 Tensor q(pd,dense(64,1,32,1)),out(pd,dense(64,32,1,1));for(int h=0;h<32;++h)for(int x=0;x<64;++x)q.put(sample(x+h),x,0,h);q.upload();out.upload();Flash4Layout f{};f.q=q.p;f.k=key.p;f.v=value.p;f.y=out.p;f.scale=.125f;Kernel flash;
 auto run=[&]{require(flash_attention(flash,pd,out.ptr(),out.bytes(),q.ptr(),q.bytes(),key.ptr(),bytes,value.ptr(),bytes,nullptr,0,f));};auto before=kv_max_statistics(pd);run();run();auto after=kv_max_statistics(pd);require(after.hits>before.hits&&after.verify_mismatches==0);
 float saved=source.get(0,300);source.put(NAN,0,300);source.upload();write(key);source.put(saved,0,300);source.upload();write(key);
 before=kv_max_statistics(pd);run();after=kv_max_statistics(pd);require(after.fallback_poison>before.fallback_poison);out.download();for(int h=0;h<32;++h)for(int x=0;x<64;++x)require(std::isfinite(out.get(x,h)));
 printf("LARGE_KV pd=%d rows=513 source_bytes=%d index_bytes=%d shuffled=1 copy_bitwise=1 verify_mismatch=0 poisoned_then_repaired_fallback=pass\n",pd,source_bytes,index_bytes);
}
int main(int argc,char **argv){try{
 setvbuf(stdout,nullptr,_IONBF,0);int pd=argc>1?std::stoi(argv[1]):0;require(pd==0||pd==1);
 if(argc>2){flash_test(pd,64,8,64,8,2,false,false);puts("SMALL_PASS");return 0;}
 flash_test(pd,64,8,64,8,2,false,false);
 flash_test(pd,128,512,512,32,32,false,true);
 flash_test(pd,64,513,768,32,4,false,true);
 flash_test(pd,64,257,544,8,2,true,false,2);
 rope_test(pd,128,32,512,InputType::f32,0);rope_test(pd,64,4,513,InputType::f16,2);kv_test(pd,4,8);kv_test(pd,2,4);
 puts("LARGE_PP_TEST_PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"LARGE_PP_TEST_FAIL %s\n",e.what());return 1;}}
