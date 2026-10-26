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
 int pd=argc>1?atoi(argv[1]):0,d=argc>2?atoi(argv[2]):64,s=argc>3?atoi(argv[3]):2048,hk=argc>4?atoi(argv[4]):4,h=32,capacity=s+256,features=d*hk;
 auto qp=dense(d,1,h,1),kp=dense(d,s,hk,1,InputType::f16);kp.nb[1]=features*2;kp.nb[2]=d*2;kp.nb[3]=features*s*2;
 auto yp=dense(d,h,1,1);Tensor q(pd,qp),y(pd,yp),gold(pd,yp),mask(pd,dense(s,1,1,1));Buffer k(pd,features*capacity*2),v(pd,features*capacity*2);
 std::vector<_Float16>kr((size_t)features*capacity),vr(kr.size());
 for(int r=0;r<capacity;++r)for(int j=0;j<features;++j){kr[(size_t)r*features+j]=cosf(j*.19f+r*.11f);vr[(size_t)r*features+j]=sinf(j*.07f+r*.03f);}
 for(int j=0;j<h;++j)for(int x=0;x<d;++x)q.put(sinf(x*.13f+j*.17f),x,0,j);
 for(int x=0;x<s;++x)mask.put(0,x);q.upload();mask.upload();
 KVMaxCache kc(pd,k.data(),k.size(),features,capacity,"unit-K"),vc(pd,v.data(),v.size(),features,capacity,"unit-V");
 auto upload=[&]{check(copy(k.data(),kr.data(),k.size(),host_to_device));check(copy(v.data(),vr.data(),v.size(),host_to_device));check(wait());};upload();
 Flash4Layout p{};p.q=qp;p.k=p.v=kp;p.y=yp;p.mask=mask.p;p.has_mask=true;p.scale=1.f/sqrtf(d);
 Kernel flash,writer;size_t offset=0;std::string decode=getenv("KUNLUN_FLASH_DECODING")?getenv("KUNLUN_FLASH_DECODING"):"auto";
 auto run=[&](const char*mode,Tensor &out){setenv("KUNLUN_FLASH_KV_MAX",mode,1);require(flash_attention(flash,pd,out.ptr(),out.bytes(),q.ptr(),q.bytes(),(char*)k.data()+offset,k.size()-offset,(char*)v.data()+offset,v.size()-offset,mask.ptr(),mask.bytes(),p));};
 auto exact=[&](const char *label){run("verify",y);run("legacy",gold);y.download();gold.download();int diff=0;for(int j=0;j<h;++j)for(int x=0;x<d;++x)diff+=memcmp(y.raw.data()+y.off(x,j),gold.raw.data()+gold.off(x,j),4)!=0;printf("KV_CASE pd=%d D=%d S=%d HK=%d case=%s bits=%d\n",pd,d,s,hk,label,diff);require(!diff);};
 exact("cold");exact("hit");
 Buffer src(pd,features*128*4),ids(pd,128*8);std::vector<float>row(features*128);std::vector<int64_t>targets(128);SetRowsLayout w{};w.width=features;w.rows=1;w.dim2=w.dim3=1;w.source_stride[0]=4;w.source_stride[1]=features*4;w.source_stride[2]=w.source_stride[3]=features*128*4;w.destination_stride[0]=features*2;w.destination_stride[1]=w.destination_stride[2]=features*capacity*2;w.index_stride[0]=8;w.index_stride[1]=w.index_stride[2]=1024;w.index_extent[0]=w.index_extent[1]=1;w.source_bytes=4;w.index_bytes=8;w.destination_rows=capacity;
 auto write=[&](const char *mode,int target,int count,float scale){w.rows=count;for(int i=0;i<count;++i)targets[i]=target+i;for(size_t j=0;j<(size_t)features*count;++j)row[j]=scale*sinf(j*.17f);check(copy(src.data(),row.data(),features*count*4,host_to_device));check(copy(ids.data(),targets.data(),count*8,host_to_device));check(wait());setenv("KUNLUN_FLASH_KV_MAX",mode,1);require(kv_max_set_rows(writer,pd,k.data(),k.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),count));};
 write("verify",s-1,1,64);exact("append-outlier");write("verify",s-1,1,.5);exact("overwrite-remove-max");
 write("verify",0,128,.25);exact("PP-writer");
 write("legacy",s/2,1,4);exact("lazy-dirty");
 kr[features*(s/3)+5]=(_Float16)-128;check(copy((char*)k.data()+2*(features*(s/3)+5),&kr[features*(s/3)+5],2,host_to_device));check(wait());exact("partial-H2D");
 upload();exact("restore");
 auto original=p.k;p.k.ne[1]=p.v.ne[1]=s-32;p.mask.ne[0]=s-32;exact("rollback-range");offset=32*features*2;exact("offset-range");offset=0;p.k=p.v=original;p.mask=mask.p;exact("grow-range");
 // F16 edge values and exceptional data use exact SDK or poison fallback.
 unsigned short edges[]={0x0001,0x03ff,0x0400,0x7bff,0x8001};for(auto bits:edges){check(copy(k.data(),&bits,2,host_to_device));check(wait());exact("F16-edge");}
 upload();exact("restore-final");
 std::vector<unsigned short>zero(kr.size());check(copy(k.data(),zero.data(),k.size(),host_to_device));check(copy(v.data(),zero.data(),v.size(),host_to_device));check(wait());kv_max_zero(pd,k.data(),k.size());kv_max_zero(pd,v.data(),v.size());exact("zero");
 for(unsigned short bits:{(unsigned short)1,(unsigned short)2,(unsigned short)0x3ff,(unsigned short)0x400,(unsigned short)0x7bff}){check(copy(k.data(),&bits,2,host_to_device));check(wait());exact("subnormal-or-finite-max");}
 // Poison is conservative: a finite overwrite does not silently clear it.
 write("verify",s-1,1,INFINITY);write("legacy",s-1,1,1);exact("nonfinite-poison-finite-overwrite");
 upload();kv_max_zero(pd,k.data(),0); // zero-length must not validate anything.
 exact("poison-restore-fallback");
 // Only a real clear or a new generation retires poison.
 check(copy(k.data(),zero.data(),k.size(),host_to_device));check(wait());kv_max_zero(pd,k.data(),k.size());upload();exact("poison-clear-rebuild");
 w.source_bytes=2;w.source_stride[0]=2;w.source_stride[1]=features*2;std::vector<_Float16>halfrow(features);for(int j=0;j<features;++j)halfrow[j]=(_Float16)(.3333333f*sinf(j*.17f));targets[0]=s-1;w.rows=1;
 check(copy(src.data(),halfrow.data(),features*2,host_to_device));check(copy(ids.data(),targets.data(),8,host_to_device));check(wait());setenv("KUNLUN_FLASH_KV_MAX","verify",1);require(kv_max_set_rows(writer,pd,k.data(),k.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),1));
 std::vector<_Float16>stored(features);check(copy(stored.data(),(char*)k.data()+(size_t)(s-1)*features*2,features*2,device_to_host));check(wait());require(!memcmp(stored.data(),halfrow.data(),features*2));exact("F16-writer");w.source_bytes=4;w.source_stride[0]=4;w.source_stride[1]=features*4;
 upload();exact("restore-before-timing");
 auto median=[](std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];};double api[2],writer_ms[2];
 for(int mode=0;mode<2;++mode){std::vector<double>at,wt;const char *m=mode?"auto":"legacy";if(mode)run("verify",y);setenv("K200_PROFILE_PHASE",mode?"timed.auto":"timed.legacy",1);run(m,y);for(int rep=0;rep<11;++rep){auto t=std::chrono::steady_clock::now();run(m,y);if(rep)at.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
  setenv("K200_PROFILE_PHASE","writer.only",1);write(m,s-1,1,1);for(int rep=0;rep<11;++rep){auto t=std::chrono::steady_clock::now();setenv("KUNLUN_FLASH_KV_MAX",m,1);require(kv_max_set_rows(writer,pd,k.data(),k.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),1));double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();if(rep)wt.push_back(ms);run(m,y);}api[mode]=median(at);writer_ms[mode]=median(wt);
 }
 printf("KV_PERF pd=%d D=%d S=%d HK=%d decode=%s api_legacy_ms=%.6f api_auto_ms=%.6f writer_legacy_ms=%.6f writer_auto_ms=%.6f total_legacy_ms=%.6f total_auto_ms=%.6f\n",pd,d,s,hk,decode.c_str(),api[0],api[1],writer_ms[0],writer_ms[1],api[0]+writer_ms[0]*2,api[1]+writer_ms[1]*2);kv_max_log_statistics();fflush(stdout);
 // The transfer/quantization contract is unchanged: compare CPU FP64 on F16 Q.
 upload();run("auto",y);y.download();double err=0,en=0,mxerr=0;
 for(int head=0;head<h;++head){std::vector<double>scores(s);double mx=-INFINITY,den=0;for(int r=0;r<s;++r){double dot=0;for(int x=0;x<d;++x)dot+=(double)(_Float16)q.get(x,0,head)*(float)kr[(size_t)r*features+(head/(h/hk))*d+x];scores[r]=dot*p.scale;mx=std::max(mx,scores[r]);}for(auto &a:scores){a=exp(a-mx);den+=a;}for(int x=0;x<d;++x){double ref=0;for(int r=0;r<s;++r)ref+=scores[r]/den*(float)vr[(size_t)r*features+(head/(h/hk))*d+x];double delta=y.get(x,head)-ref;err+=delta*delta;en+=ref*ref;mxerr=std::max(mxerr,fabs(delta));}}
 printf("KV_CPU pd=%d D=%d S=%d HK=%d nmse=%.12g max=%.12g\n",pd,d,s,hk,err/std::max(en,1e-30),mxerr);require(err/std::max(en,1e-30)<2e-4);
 double append_ms[2];
 for(int mode=0;mode<2;++mode){
  setenv("K200_PROFILE_PHASE",mode?"append.auto":"append.legacy",1);
  check(copy(k.data(),zero.data(),k.size(),host_to_device));check(copy(v.data(),zero.data(),v.size(),host_to_device));check(wait());kv_max_zero(pd,k.data(),k.size());kv_max_zero(pd,v.data(),v.size());
  // Identical finite prefix for both arms, with 16 proven-zero append slots.
  for(int first=0;first<s-16;first+=128){int n=std::min(128,s-16-first);write("auto",first,n,1);setenv("KUNLUN_FLASH_KV_MAX","auto",1);require(kv_max_set_rows(writer,pd,v.data(),v.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),n));}
  run(mode?"auto":"legacy",y);std::vector<double>times;w.rows=1;
  for(int j=0;j<16;++j){targets[0]=s-16+j;check(copy(ids.data(),targets.data(),8,host_to_device));check(wait());const char *md=mode?"auto":"legacy";setenv("KUNLUN_FLASH_KV_MAX",md,1);auto t=std::chrono::steady_clock::now();
   require(kv_max_set_rows(writer,pd,k.data(),k.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),1));require(kv_max_set_rows(writer,pd,v.data(),v.size(),src.data(),src.size(),ids.data(),ids.size(),w,targets.data(),1));run(md,y);times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
  }append_ms[mode]=median(times);exact(mode?"append-sequence-auto":"append-sequence-legacy");
 }
 printf("KV_APPEND pd=%d D=%d S=%d HK=%d decode=%s legacy_ms=%.6f auto_ms=%.6f speedup=%.6f includes_writers=2 includes_api=1 input_upload_outside=1\n",pd,d,s,hk,decode.c_str(),append_ms[0],append_ms[1],append_ms[0]/append_ms[1]);
}
