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
#include <xpu/runtime.h>
#include "flash-config.h"
#include <fstream>
struct Arg{const void*p;size_t n;};template<class T>Arg arg(const T&x){return {&x,sizeof(x)};}
static void launch(XPUFunc fn,std::vector<Arg> args){size_t off=0;for(auto a:args){check(xpu_launch_argument_set(a.p,a.n,off));off+=a.n;}check(xpu_launch_config(4,8));check(xpu_launch_async(fn));}
static double elapsed(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char**argv){
 if(argc<8)return 2;
 int pd=atoi(argv[1]),d=atoi(argv[2]),s=atoi(argv[3]),m=atoi(argv[4]),hk=atoi(argv[5]),h=atoi(argv[6]);std::string dir=argv[7];int pattern=argc>8?atoi(argv[8]):0,batch=argc>9?atoi(argv[9]):1,qt=argc>10?atoi(argv[10]):128;
 if(pd<0||pd>1||(d!=64&&d!=128)||s<32||s%32||m<1||m>256||hk<1||h<hk||h>128||h%hk||batch<1||batch>4||qt<16||qt>128||qt%16)return 2;
 if(pattern==18&&batch%2)return 2;
 int block=std::min(s,m==1?1024:512),s0=s-block,mt=(m+qt-1)/qt,rows=m*h*batch,kvbatch=pattern==18?2:1;
 auto qp=dense(d,m,h,batch,InputType::f16,pattern==3?32:0),kp=dense(d,s,hk,kvbatch,InputType::f16,pattern==3?32:0);
 if(pattern==11){qp.nb[1]=d*h*2;qp.nb[2]=d*2;qp.nb[3]=d*h*m*2;kp.nb[1]=d*hk*2;kp.nb[2]=d*2;kp.nb[3]=d*hk*s*2;}
 Tensor q(pd,qp),k(pd,kp);
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x){float val=sinf(x*.13f+r*.07f+j*.17f+b*.3f);if(pattern==17)val=j==h-1?(x==d-1?60000.f:0.f):.001f*sinf(x*.17f+j);if(pattern==8)val=0;if(pattern==9)val*=1e-7f;q.put(val,x,r,j,b);}
 for(int b=0;b<kvbatch;++b)for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x)k.put(cosf(x*.19f+r*.11f+j*.2f+b*.7f)*(pattern==17?6000.f:1.f),x,r,j,b);
 if(pattern==4){q.put(32,d-1,m-1,h-1,batch-1);k.put(-16,d-1,s-1,hk-1,kvbatch-1);}
 float peak=0;for(int b=0;b<kvbatch;++b)for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x)peak=std::max(peak,fabsf(k.get(x,r,j,b)));
 std::vector<float>range((size_t)batch*h*mt*8);for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int mi=0;mi<mt;++mi){float v=0;for(int r=mi*qt;r<std::min(m,(mi+1)*qt);++r)for(int x=0;x<d;++x)v=std::max(v,fabsf(q.get(x,r,j,b)));range[((b*h+j)*mt+mi)*8]=v;}
 std::vector<float>peaks(64);peaks[0]=peak*.1f;peaks[1]=peak*.2f;peaks[2]=peak*.3f;peaks[3]=peak;
 KunlunFlashConfig p{};p.d=d;p.s=s;p.block=block;p.m=m;p.h=h;p.hk=hk;p.b=batch;p.bk=p.bv=kvbatch;p.qt=qt;p.tg_group=1;std::copy(qp.nb,qp.nb+4,p.q);std::copy(kp.nb,kp.nb+4,p.k);
 Buffer pc(pd,256),pk(pd,256),pr(pd,range.size()*4),diag(pd,256);check(copy(pc.data(),&p,sizeof(p),host_to_device));check(copy(pk.data(),peaks.data(),256,host_to_device));check(copy(pr.data(),range.data(),range.size()*4,host_to_device));q.upload();k.upload();auto qr=q.raw,kr=k.raw;
 size_t bytes=(size_t)rows*block*4;if(bytes+256>16*1024*1024){fprintf(stderr,"standalone score exceeds L3 capacity\n");return 2;}void *score=nullptr;check(xpu_malloc(&score,bytes+256,XPU_MEM_L3));require(score);std::vector<unsigned char>init(bytes+256,0xa5);
 std::vector<std::string>modes={"","-reuse1024"};if(argc>11){std::string selected=argv[11];modes=selected.find("profile")!=std::string::npos?std::vector<std::string>{selected}:std::vector<std::string>{"",selected};}
 std::vector<float>ref;int failure=0;
 for(const auto&mode:modes){
  std::string path=dir+"/kunlun-flash-kq"+mode;std::ifstream bin(path+".bin",std::ios::binary),entry(path+".entry");std::vector<char>code((std::istreambuf_iterator<char>(bin)),{});std::string symbol;getline(entry,symbol);require(!code.empty()&&!symbol.empty());code.resize(((code.size()+255)&~size_t(255))+256);require(code.size()<=16384);
  Buffer program(pd,code.size());check(copy(program.data(),code.data(),code.size(),host_to_device));check(wait());XPUFunc fn;check(xpu_create_sd_func(&fn,(uint64_t)(uintptr_t)program.data(),code.size(),0,0,0,symbol.c_str(),true));
  const void *qp_=q.ptr(),*kp_=k.ptr(),*cp=pc.data(),*pkp=pk.data(),*prp=pr.data();void *dp=diag.data();bool profile=mode.find("profile")!=std::string::npos;
  std::vector<double>times,enq;check(copy(score,init.data(),init.size(),host_to_device));check(wait());
  for(int rep=0;rep<13;++rep){auto t=std::chrono::steady_clock::now();std::vector<Arg>a={arg(score),arg(kp_),arg(qp_),arg(pkp),arg(prp),arg(cp),arg(s0),arg(block)};if(profile)a.push_back(arg(dp));launch(fn,a);double e=elapsed(t);check(wait());double ms=elapsed(t);if(rep){times.push_back(ms);enq.push_back(e);}}
  std::vector<float>out(bytes/4);check(copy(out.data(),score,bytes,device_to_host));std::vector<unsigned char>guard(256);check(copy(guard.data(),(char*)score+bytes,256,device_to_host));check(wait());
  if(ref.empty())ref=out;Error old,cpu;int bits=0;for(size_t i=0;i<out.size();++i){old.add(out[i],ref[i]);bits+=memcmp(&out[i],&ref[i],4)!=0;}
  for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r){if(m>4&&r!=0&&r!=m/3&&r!=2*m/3&&r!=m-1)continue;float scale=(m==1&&h>hk)?range[((b*h+j)*mt)*8]:1.f;if(scale<=0)scale=1;for(int z=0;z<block;++z){double gold=0;for(int x=0;x<d;++x)gold+=(double)q.get(x,r,j,b)/scale*k.get(x,s0+z,j/(h/hk),b/(batch/kvbatch));cpu.add(out[((b*h+j)*m+r)*block+z],gold);}}
  q.download();k.download();bool same=q.raw==qr&&k.raw==kr;int badguard=0;for(auto c:guard)badguard+=c!=0xa5;
  std::sort(times.begin(),times.end());std::sort(enq.begin(),enq.end());bool pass=!old.bad&&!cpu.bad&&!bits&&same&&!badguard&&cpu.nmse()<2e-4;
  printf("KQ_UNIT pd=%d D=%d S=%d block=%d M=%d HK=%d H=%d batch=%d qt=%d pattern=%d mode=%s ms=%.6f min_ms=%.6f enqueue_ms=%.6f bits=%d old_nmse=%.12g old_max=%.12g cpu_nmse=%.12g cpu_max=%.12g input_same=%d guard=%d pass=%d\n",pd,d,s,block,m,hk,h,batch,qt,pattern,mode.empty()?"legacy":mode.c_str()+1,times[times.size()/2],times[0],enq[enq.size()/2],bits,old.nmse(),old.max,cpu.nmse(),cpu.max,same,badguard,pass);fflush(stdout);if(!pass)++failure;
  if(profile){unsigned int ticks[64];check(copy(ticks,diag.data(),sizeof(ticks),device_to_host));check(wait());for(int cl=0;cl<4;++cl){printf("KQ_STAGE cluster=%d mode=%s",cl,mode.c_str()+1);for(int j=0;j<16;++j)printf(" t%d=%u",j,ticks[cl*16+j]);printf("\n");}}
 }
 check(xpu_free(score));return failure?1:0;
}
