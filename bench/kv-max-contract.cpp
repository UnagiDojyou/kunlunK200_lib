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
#include <memory>
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
 if(argc<7)return 2;
 int pd=atoi(argv[1]),d=atoi(argv[2]),s=atoi(argv[3]),m=atoi(argv[4]),hk=atoi(argv[5]),h=atoi(argv[6]),pattern=argc>7?atoi(argv[7]):0,batch=argc>8?atoi(argv[8]):1;
 void *reserved_l3=nullptr;if(auto hold=getenv("KUNLUN_FLASH_DECODE_BENCH_RESERVE_L3_MB")){int mb=atoi(hold);if(mb<1||mb>14)return 2;check(xpu_set_device(pd));check(xpu_malloc(&reserved_l3,(size_t)mb*1024*1024,XPU_MEM_L3));}
 auto qp=dense(d,m,h,batch);auto kp=dense(d,s,hk,1);kp.type=InputType::f16;for(int z=0;z<4;++z)kp.nb[z]/=2;
 if(pattern==18){kp.ne[3]=2;}
 auto vp=kp;auto out=dense(d,h,m,batch);auto maskp=dense(s,m,pattern==13?2:1,pattern==13?batch:1);
 if(pattern==12||pattern==25||pattern==29){maskp.type=InputType::f16;for(int z=0;z<4;++z)maskp.nb[z]/=2;}
 if(pattern==3){out.nb[1]+=32;out.nb[2]=out.nb[1]*h+32;out.nb[3]=out.nb[2]*m+32;qp.nb[1]+=32;qp.nb[2]=qp.nb[1]*m+32;qp.nb[3]=qp.nb[2]*h+32;kp.nb[1]+=32;kp.nb[2]=kp.nb[1]*s+32;kp.nb[3]=kp.nb[2]*hk+32;vp=kp;}
 if((pattern==11||pattern==29)){qp.nb[1]=d*h*4;qp.nb[2]=d*4;qp.nb[3]=d*h*m*4;kp.nb[1]=d*hk*2;kp.nb[2]=d*2;kp.nb[3]=d*hk*s*2;vp=kp;}
 if(pattern==5){qp.type=InputType::f16;for(int z=0;z<4;++z)qp.nb[z]/=2;}
 Tensor q(pd,qp),k(pd,kp),v(pd,vp),y(pd,out),mask(pd,maskp);
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)q.put(sinf(x*.13f+r*.07f+j*.17f+b*.3f)*(pattern==2?4.f:1.f),x,r,j,b);
 for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x){k.put(cosf(x*.19f+(pattern==23?r%1024:r)*.11f+j*.2f),x,r,j);v.put(sinf(x*.07f+r*.03f+j*.3f),x,r,j);}
 if(pattern==8||pattern==9){for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)q.put(pattern==8?0.f:1e-9f,x,r,j,b);}
 if(pattern==4){q.put(32,d-1,m-1,h-1,batch-1);k.put(-16,d-1,s-1,hk-1);v.put(8,d-1,s-1,hk-1);}
 for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put((pattern!=1&&pattern!=20&&x>s-m+r)?-INFINITY:((pattern==6&&r==0)?-INFINITY:0.f),x,r);
 if(pattern==10){for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>s-m+r?-INFINITY:10000.f+(x%3),x,r);}
 if(pattern==12||pattern==13){for(int b=0;b<maskp.ne[3];++b)for(int j=0;j<maskp.ne[2];++j)for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>s-m+r?-INFINITY:(x%7)*(.01f*j+.005f*b),x,r,j,b);}
 if(pattern==14){auto random=[](uint32_t u){u=u*747796405u+2891336453u;u=((u>>((u>>28)+4))^u)*277803737u;u=(u>>22)^u;return (float)(u&0xffffff)/8388608.f-1.f;};for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)q.put(random(x+d*(r+m*(j+h*b)))*3,x,r,j,b);for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x){k.put(random(1234+x+d*(r+s*j)),x,r,j);v.put(random(987654+x+d*(r+s*j)),x,r,j);}}
 if(pattern==16){for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x){k.put(0,x,r,j);v.put(1,x,r,j);}}
 if(pattern==17){for(int j=0;j<h;++j)for(int x=0;x<d;++x)q.put(j==h-1?(x==d-1?60000.f:0.f):.001f*sinf(x*.17f+j),x,0,j);for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x)k.put(6000.f*cosf(x*.19f+(pattern==23?r%1024:r)*.11f+j*.2f),x,r,j);}
 if(pattern==18){for(int ba=0;ba<kp.ne[3];++ba)for(int j=0;j<hk;++j)for(int r=0;r<s;++r)for(int x=0;x<d;++x){k.put(cosf(x*.19f+r*.11f+j*.2f+ba*.7f),x,r,j,ba);v.put(sinf(x*.07f+r*.03f+j*.3f+ba*.9f),x,r,j,ba);}}
 if(pattern==24||pattern==25){for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>s-m+r?-INFINITY:-(s-1-x)*.01f,x,r);}
 if(pattern==27||pattern==28){for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put((pattern==27?(x>=s/4&&x<s/2):(x>=s/2))?-INFINITY:0.f,x,r);}
 if(pattern==29)for(int r=0;r<m;++r)for(int x=0;x<s;++x)mask.put(x>=s-255?-INFINITY:0.f,x,r);
 auto qr=q.raw,kr=k.raw,vr=v.raw,mr=mask.raw;q.upload();k.upload();v.upload();mask.upload();y.upload();
 std::unique_ptr<KVMaxCache> kmcache,vmcache;
 if(Tensor::span(kp)==(size_t)d*s*hk*2&&kp.ne[3]==1){kmcache.reset(new KVMaxCache(pd,k.ptr(),(size_t)d*s*hk*2,d*hk,s,"contract-K"));vmcache.reset(new KVMaxCache(pd,v.ptr(),(size_t)d*s*hk*2,d*hk,s,"contract-V"));}
 std::string maxmode=getenv("KUNLUN_FLASH_KV_MAX")?getenv("KUNLUN_FLASH_KV_MAX"):"verify";setenv("KUNLUN_FLASH_KV_MAX",maxmode.c_str(),1);

 Flash4Layout p{};p.q=qp;p.k=kp;p.v=vp;p.y=out;p.mask=maskp;p.has_mask=true;p.scale=1.f/sqrtf(d);if(pattern==20){p.has_mask=false;memset(&p.mask,0x5a,sizeof(p.mask));}if(pattern==21)p.scale=0;if(pattern==22)p.scale=-p.scale;if(pattern==7||pattern==24||pattern==25)p.max_bias=8;
 if(*flash_attention_reason(p)){printf("REJECT %s\n",flash_attention_reason(p));return 3;}
 std::string mode=getenv("KUNLUN_FLASH_DECODING")?getenv("KUNLUN_FLASH_DECODING"):"auto";Kernel flash,baseline;auto run=[&]{return flash_attention(flash,pd,y.ptr(),y.bytes(),q.ptr(),q.bytes(),k.ptr(),k.bytes(),v.ptr(),v.bytes(),mask.ptr(),mask.bytes(),p);};
 std::vector<double>times;for(int rep=0;rep<5;++rep){setenv("K200_PROFILE_PHASE",rep?"flash.steady":"flash.cold",1);auto t=std::chrono::steady_clock::now();require(run());double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();if(rep)times.push_back(ms);}
 std::sort(times.begin(),times.end());setenv("K200_PROFILE_PHASE","reference",1);y.download();
 Tensor yo(pd,out);
 auto oldrun=[&]{setenv("KUNLUN_FLASH_KV_MAX","legacy",1);require(flash_attention(baseline,pd,yo.ptr(),yo.bytes(),q.ptr(),q.bytes(),k.ptr(),k.bytes(),v.ptr(),v.bytes(),mask.ptr(),mask.bytes(),p));setenv("KUNLUN_FLASH_KV_MAX",maxmode.c_str(),1);};
 std::vector<double>ot;for(int rep=0;rep<5;++rep){auto t=std::chrono::steady_clock::now();oldrun();if(rep)ot.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}std::sort(ot.begin(),ot.end());yo.download();
 int bitdiff=0;for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)bitdiff+=memcmp(y.raw.data()+y.off(x,j,r,b),yo.raw.data()+yo.off(x,j,r,b),4)!=0;
 require(bitdiff==0);printf("KV_COMPARE mode=%s bitdiff=%d\n",maxmode.c_str(),bitdiff);
 double err=0,ref=0,maxerr=0,oe=0,orr=0,om=0;int bad=0;bool sampled=std::getenv("KUNLUN_FLASH_BENCH_SAMPLE")!=nullptr;
 for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r){
  if(sampled&&r!=0&&r!=m/3&&r!=2*m/3&&r!=m-1)continue;
  std::vector<double>l(s);double mx=-INFINITY,sum=0;
  float slope=1;if(p.max_bias>0){int nh=1;while(nh*2<=h)nh*=2;float b0=powf(2,-p.max_bias/nh),b1=powf(2,-p.max_bias/(2*nh));slope=j<nh?powf(b0,j+1):powf(b1,2*(j-nh)+1);}
  for(int z=0;z<s;++z){double dot=0;for(int x=0;x<d;++x)dot+=(double)(getenv("KUNLUN_FLASH_FULL_Q_REF")?q.get(x,r,j,b):(float)(_Float16)q.get(x,r,j,b))*k.get(x,z,j/(h/hk),b/(batch/kp.ne[3]));l[z]=dot*p.scale+mask.get(z,r,j%maskp.ne[2],b%maskp.ne[3])*slope;if(l[z]>mx)mx=l[z];}
  for(int z=0;z<s;++z){l[z]=l[z]==-INFINITY?0:exp(l[z]-mx);sum+=l[z];}
  for(int x=0;x<d;++x){double gold=0;if(sum>0)for(int z=0;z<s;++z)gold+=l[z]/sum*v.get(x,z,j/(h/hk),b/(batch/vp.ne[3]));double got=y.get(x,j,r,b),old=yo.get(x,j,r,b),e=got-gold,delta=got-old;err+=e*e;ref+=gold*gold;maxerr=std::max(maxerr,fabs(e));oe+=delta*delta;orr+=old*old;om=std::max(om,fabs(delta));if(!std::isfinite(got))++bad;}
 }
 Error full_old;for(int ba=0;ba<batch;++ba)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)full_old.add(y.get(x,j,r,ba),yo.get(x,j,r,ba));oe=full_old.sq;orr=full_old.en;om=full_old.max;bad+=full_old.bad;
 q.download();k.download();v.download();mask.download();bool same=q.raw==qr&&k.raw==kr&&v.raw==vr&&mask.raw==mr;bool pass=!bad&&same&&y.guards()==0&&err/(ref+1e-30)<2e-4;
 auto fs=flash_statistics(flash);std::string block_label=std::to_string(fs.block),qt_label=std::to_string(fs.query_tile);
 printf("FLASH_UNIT pd=%d D=%d S=%d M=%d HK=%d H=%d batch=%d pattern=%d block=%s storage=%s flash_ms=%.6f old_ms=%.6f cpu_nmse=%.12g cpu_max=%.12g old_nmse=%.12g old_max=%.12g input_same=%d guard=%d pass=%d qt=%s sampled=%d full_q=%d old_full=1\n",pd,d,s,m,hk,h,batch,pattern,block_label.c_str(),fs.l3_workspace_bytes?"L3":"HBM",times[times.size()/2],ot[ot.size()/2],err/(ref+1e-30),maxerr,oe/(orr+1e-30),om,same,y.guards(),pass,qt_label.c_str(),sampled,getenv("KUNLUN_FLASH_FULL_Q_REF")!=nullptr);fflush(stdout);
 if(getenv("KUNLUN_FLASH_BENCH_MUTATE")){
  q.put(4,d-1,m-1,h-1,batch-1);k.put(-8,d-1,s-1,hk-1);v.put(16,d-1,s-1,hk-1);
  q.upload();k.upload();v.upload();auto qm=q.raw,km=k.raw,vm=v.raw;
  require(run());oldrun();y.download();yo.download();Error delta;
  for(int b=0;b<batch;++b)for(int j=0;j<h;++j)for(int r=0;r<m;++r)for(int x=0;x<d;++x)delta.add(y.get(x,j,r,b),yo.get(x,j,r,b));
  q.download();k.download();v.download();require(!delta.bad&&delta.nmse()<2e-4&&q.raw==qm&&k.raw==km&&v.raw==vm);
  printf("FLASH_MUTATE pd=%d D=%d S=%d M=%d old_nmse=%.12g max_abs=%.12g input_same=1 pass=1\n",pd,d,s,m,delta.nmse(),delta.max);
 }
 if(reserved_l3)check(xpu_free(reserved_l3));
 kv_max_log_statistics();return pass?0:1;
}
