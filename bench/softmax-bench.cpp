#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
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
int main(int argc,char**argv){
 if(argc!=13){fprintf(stderr,"usage: %s pd N Q heads batches input-bytes output-bytes mask-bytes pattern gap permute inplace\n",argv[0]);return 2;}
 int pd=atoi(argv[1]),n=atoi(argv[2]),q=atoi(argv[3]),heads=atoi(argv[4]),batches=atoi(argv[5]),xb=atoi(argv[6]),yb=atoi(argv[7]),mb=atoi(argv[8]),pat=atoi(argv[9]),gap=atoi(argv[10]),perm=atoi(argv[11]),alias=atoi(argv[12]);
 require(pd>=0&&pd<2&&n>0&&n<=16384&&q>0&&heads>0&&batches>0&&(xb==2||xb==4)&&(yb==2||yb==4)&&(mb==0||mb==2||mb==4)&&(!alias||xb==yb));
 auto xt=xb==2?InputType::f16:InputType::f32,yt=yb==2?InputType::f16:InputType::f32;
 auto xl=dense(n,q,heads,batches,xt,gap),yl=dense(n,q,heads,batches,yt,gap+8);
 if(perm){xl.nb[2]=n*xb+gap;xl.nb[1]=xl.nb[2]*heads;xl.nb[3]=xl.nb[1]*q;yl.nb[2]=n*yb+gap+8;yl.nb[1]=yl.nb[2]*heads;yl.nb[3]=yl.nb[1]*q;}
 if(alias)yl=xl;
 Tensor x(pd,xl),y(pd,yl),mask(pd,dense(n,q+1,heads%2==0?2:1,1,mb==2?InputType::f16:InputType::f32,gap));
 for(int b=0;b<batches;++b)for(int h=0;h<heads;++h)for(int r=0;r<q;++r)for(int c=0;c<n;++c){float v=sample(c+r*7+h*11+b*17)*8;
  if(pat==4)v=(c%3-1)*(xb==2?60000.f:1e30f);if(pat==5)v=c==0?0:-80.f-c%40;if(pat==6)v=1000;
  if(pat==7)v=c==0?INFINITY:(c==1?-INFINITY:v);if(pat==8)v=c==0?NAN:v;if(pat==9)v=NAN;if(pat==10)v=-INFINITY;
  x.put(v,c,r,h,b);
 }
 for(int h=0;h<mask.p.ne[2];++h)for(int r=0;r<=q;++r)for(int c=0;c<n;++c){float v=-(c%5)*.125f*(h+1);if(pat==1&&c>n/2+r)v=-INFINITY;if(pat==2&&(r==0||c%3==0))v=-INFINITY;if(pat==3&&r%2==0)v=-INFINITY;mask.put(v,c,r,h);}
 auto original=x.raw,originalmask=mask.raw;x.upload();y.upload();mask.upload();
 Softmax4Layout sm{xl,yl,mask.p,mb!=0,pat==5?1.f:pat==11?0.f:pat==12?-.125f:.125f,pat==13?8.f:0.f};
 std::vector<float>ref((size_t)n*q*heads*batches);size_t idx=0;
 for(int b=0;b<batches;++b)for(int h=0;h<heads;++h)for(int r=0;r<q;++r){float slope=1;if(sm.max_bias>0){int nh=1;while(nh*2<=heads)nh*=2;float b0=std::pow(2.f,-sm.max_bias/nh),b1=std::pow(2.f,-sm.max_bias/(2*nh));slope=h<nh?std::pow(b0,float(h+1)):std::pow(b1,float(2*(h-nh)+1));}
  std::vector<float>v(n);float mx=-INFINITY;for(int c=0;c<n;++c){v[c]=x.get(c,r,h,b)*sm.scale+(mb?slope*mask.get(c,r,h%mask.p.ne[2]):0);if(v[c]>mx)mx=v[c];}
  double sum=0;for(int c=0;c<n;++c)sum+=mx==-INFINITY?0:std::exp(double(v[c])-mx);
  for(int c=0;c<n;++c){float z=mx==-INFINITY?0:std::exp(double(v[c])-mx)*(sum>0?1/sum:0);ref[idx++]=yb==2?float((_Float16)z):z;}
 }
 std::vector<float>old;int pass=1;
 for(const char*mode:{"legacy","fast"}){
  if((xb==2||yb==2)&&strcmp(mode,"legacy")&&strcmp(mode,"fast"))continue;
  setenv("KUNLUN_ATTENTION_SOFTMAX",mode,1);Kernel kernel;void*out=alias?x.ptr():y.ptr();size_t bytes=alias?x.bytes():y.bytes();
  x.raw=original;x.upload();auto run=[&]{return attention_softmax(kernel,pd,out,bytes,x.ptr(),x.bytes(),mb?mask.ptr():nullptr,mb?mask.bytes():0,sm);};require(run());
  std::vector<double>times;for(int t=0;t<(alias?1:7);++t){if(alias){x.raw=original;x.upload();}auto start=std::chrono::steady_clock::now();require(run());times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
  Tensor&z=alias?x:y;z.download();Error cpu,delta;int classes=0,oldclasses=0,changed=0;idx=0;std::vector<float>values;
  for(int b=0;b<batches;++b)for(int h=0;h<heads;++h)for(int r=0;r<q;++r)for(int c=0;c<n;++c){float a=z.get(c,r,h,b),v=ref[idx];values.push_back(a);if(std::isfinite(a)&&std::isfinite(v))cpu.add(a,v);else if(!(std::isnan(a)&&std::isnan(v))&&a!=v){++classes;static int shown=0;if(shown++<3)printf("SPECIAL actual=%g cpu=%g pat=%d mode=%s\n",a,v,pat,mode);}if(!old.empty()){float o=old[idx];if(std::isfinite(a)&&std::isfinite(o))delta.add(a,o);else if(!(std::isnan(a)&&std::isnan(o))&&a!=o)++oldclasses;if(memcmp(&a,&o,4))++changed;}++idx;}
  if(old.empty())old=values;
  int guards=z.guards();if(!alias){x.download();guards+=x.raw!=original;}mask.download();guards+=mask.raw!=originalmask;
  std::sort(times.begin(),times.end());int ok=oldclasses==0&&(classes==0||(pat>=7&&pat<=9))&&guards==0&&cpu.nmse()<(yb==2?2e-6:1e-9)&&delta.nmse()<1e-9;pass&=ok;
  printf("SOFTMAX pd=%d n=%d q=%d h=%d b=%d xb=%d yb=%d mb=%d pat=%d gap=%d perm=%d alias=%d mode=%s ms=%.6f cpu_nmse=%.12g cpu_max=%.12g old_nmse=%.12g old_max=%.12g changed=%d classes=%d old_classes=%d guards=%d pass=%d\n",pd,n,q,heads,batches,xb,yb,mb,pat,gap,perm,alias,mode,times[times.size()/2],cpu.nmse(),cpu.max,delta.nmse(),delta.max,changed,classes,oldclasses,guards,ok);fflush(stdout);
 }
 return pass?0:1;
}
