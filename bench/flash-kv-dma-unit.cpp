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

#include <memory>
struct Program {
 std::string symbol;std::unique_ptr<Buffer> code;XPUFunc fn;
 Program(int pd,std::string path):code(){std::ifstream bin(path+".bin",std::ios::binary),entry(path+".entry");std::vector<char> bytes((std::istreambuf_iterator<char>(bin)),{});getline(entry,symbol);require(!bytes.empty()&&!symbol.empty());bytes.resize(((bytes.size()+255)&~size_t(255))+256);require(bytes.size()<=16384);code.reset(new Buffer(pd,bytes.size()));check(copy(code->data(),bytes.data(),bytes.size(),host_to_device));check(wait());check(xpu_create_sd_func(&fn,(uint64_t)(uintptr_t)code->data(),bytes.size(),0,0,0,symbol.c_str(),true));}
};
int main(int argc,char **argv){
 if(argc<10)return 2;int pd=atoi(argv[1]),d=atoi(argv[2]),s=atoi(argv[3]),m=atoi(argv[4]),hk=atoi(argv[5]),h=atoi(argv[6]),layout=atoi(argv[8]);std::string dir=argv[7],op=argv[9];int batch=argc>10?atoi(argv[10]):1;
 require((d==64||d==128)&&s%32==0&&s>=32&&hk>0&&h%hk==0&&(m==1||m==128));select(pd);
 int block=std::min(s,m==1?1024:512),s0=s-block,mt=(m+127)/128,rows=m*h*batch;bool pv=op=="pv";
 auto kp=dense(d,s,hk,batch,InputType::f16,layout==2?32:0),qp=dense(pv?block:d,m,h,batch,pv?InputType::f32:InputType::f16,0);
 if(layout==1){kp.nb[1]=d*hk*2;kp.nb[2]=d*2;kp.nb[3]=d*hk*s*2;}
 Tensor k(pd,kp),q(pd,qp);
 for(int b=0;b<batch;++b)for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x)k.put(cosf(x*.19f+r*.11f+j*.2f+b*.7f),x,r,j,b);
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<qp.ne[0];++x)q.put(pv?(1+sinf(x*.13f+r*.07f+j*.17f+b*.3f))*.5f/block:sinf(x*.13f+r*.07f+j*.17f+b*.3f),x,r,j,b);
 float peak=0;for(int b=0;b<batch;++b)for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x)peak=std::max(peak,fabsf(k.get(x,r,j,b)));
 std::vector<float> range((size_t)batch*h*mt*8),peaks(64);peaks[0]=.1f*peak;peaks[1]=.2f*peak;peaks[2]=.3f*peak;peaks[3]=peak;
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j){float v=0;for(int r=0;r<m;++r)for(int x=0;x<qp.ne[0];++x)v=std::max(v,fabsf(q.get(x,r,j,b)));range[(b*h+j)*mt*8]=v;}
 KunlunFlashConfig p{};p.d=d;p.s=s;p.m=m;p.h=h;p.hk=hk;p.b=p.bk=p.bv=batch;p.qt=128;p.block=block;p.tg_group=1;std::copy(kp.nb,kp.nb+4,p.k);std::copy(kp.nb,kp.nb+4,p.v);std::copy(qp.nb,qp.nb+4,p.q);
 Buffer pc(pd,256),pk(pd,256),pr(pd,range.size()*4);check(copy(pc.data(),&p,sizeof(p),host_to_device));check(copy(pk.data(),peaks.data(),256,host_to_device));check(copy(pr.data(),range.data(),range.size()*4,host_to_device));q.upload();k.upload();auto qr=q.raw,kr=k.raw;
 size_t bytes=(size_t)rows*(pv?d:block)*4;require(bytes+256<=16*1024*1024);void *out=nullptr;check(xpu_malloc(&out,bytes+256,XPU_MEM_L3));std::vector<unsigned char>init(bytes+256,0xa5);check(copy(out,init.data(),init.size(),host_to_device));check(wait());
 Program old(pd,dir+(pv?"/kunlun-flash-pv-wide":"/kunlun-flash-kq-reuse1024")),candidate(pd,dir+(pv?"/kunlun-flash-pv-kv-group":"/kunlun-flash-kq-kv-group"));
 const void *kp_=k.ptr(),*qp_=q.ptr(),*cp=pc.data(),*pkp=pk.data(),*prp=pr.data();
 auto invoke=[&](bool newer,int g){std::vector<Arg>a={arg(out),arg(kp_),arg(qp_),arg(pkp)};if(!pv)a.push_back(arg(prp));a.push_back(arg(cp));a.push_back(arg(s0));a.push_back(arg(block));if(newer)a.push_back(arg(g));launch(newer?candidate.fn:old.fn,a);};
 std::vector<float>ref(bytes/4),got(bytes/4);invoke(false,1);check(wait());check(copy(ref.data(),out,bytes,device_to_host));check(wait());Error cpu;
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r){if(m>4&&r!=0&&r!=m/3&&r!=2*m/3&&r!=m-1)continue;for(int x=0;x<(pv?d:block);++x){double gold=0;for(int z=0;z<(pv?block:d);++z)gold+=pv?(double)q.get(z,r,j,b)*k.get(x,s0+z,j/(h/hk),b):(double)q.get(z,r,j,b)*k.get(z,s0+x,j/(h/hk),b);if(!pv&&m==1&&h>hk)gold/=range[(b*h+j)*mt*8];cpu.add(ref[((b*h+j)*m+r)*(pv?d:block)+x],gold);}}
 int failure=cpu.bad;std::vector<int>gs={1};if(layout==1)for(int g:{2,4,8})if(hk%g==0&&(size_t)block*d*g*2<=2*1024*1024)gs.push_back(g);
 if(getenv("KV_GROUP")){gs={atoi(getenv("KV_GROUP"))};require(gs[0]>=1&&hk%gs[0]==0);}
 for(int g:gs){std::vector<double>times[2],enq[2];int bits=0,badguard=0;Error diff;
  for(int rep=-4;rep<24;++rep){int idx=(rep%4+4)%4;bool newer=idx==1||idx==2;auto t=std::chrono::steady_clock::now();invoke(newer,g);double en=elapsed(t);check(wait());double ms=elapsed(t);if(rep>=0){times[newer].push_back(ms);enq[newer].push_back(en);}check(copy(got.data(),out,bytes,device_to_host));check(wait());for(size_t i=0;i<got.size();++i){if(memcmp(&got[i],&ref[i],4))++bits;diff.add(got[i],ref[i]);}}
  std::vector<unsigned char>guard(256);check(copy(guard.data(),(char*)out+bytes,256,device_to_host));check(wait());for(auto v:guard)badguard+=v!=0xa5;q.download();k.download();bool same=q.raw==qr&&k.raw==kr;bool pass=!bits&&!diff.bad&&!badguard&&same&&!cpu.bad;failure+=!pass;
  for(int n=0;n<2;++n){std::sort(times[n].begin(),times[n].end());std::sort(enq[n].begin(),enq[n].end());printf("KV_UNIT op=%s pd=%d D=%d S=%d B=%d M=%d HK=%d H=%d batch=%d layout=%d group=%d mode=%s ms=%.6f min_ms=%.6f enqueue_ms=%.6f bits=%d nmse=%.12g max=%.12g cpu_nmse=%.12g cpu_max=%.12g same=%d guard=%d pass=%d\n",op.c_str(),pd,d,s,block,m,hk,h,batch,layout,g,n?"group":"legacy",times[n][times[n].size()/2],times[n][0],enq[n][enq[n].size()/2],bits,diff.nmse(),diff.max,cpu.nmse(),cpu.max,same,badguard,pass);fflush(stdout);}
 }
 check(xpu_free(out));return failure?1:0;
}
