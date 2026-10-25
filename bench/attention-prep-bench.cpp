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
int main(int argc,char**argv){
 if(argc!=13){fprintf(stderr,"usage: %s pd op D S M HK HQ B-bytes pattern gap batches layout\n",argv[0]);return 2;}int pd=atoi(argv[1]),op=atoi(argv[2]),d=atoi(argv[3]),s=atoi(argv[4]),m=atoi(argv[5]),hk=atoi(argv[6]),hq=atoi(argv[7]),xb=atoi(argv[8]),pattern=atoi(argv[9]),gap=atoi(argv[10]),batch=atoi(argv[11]),layout=atoi(argv[12]);
 int k=op?s:d,n=op?d:s;auto bt=xb==2?InputType::f16:InputType::f32;
 auto al=dense(k,n,hk,1,InputType::f16,gap);if(op){al.nb[1]=(k+(layout==2?17:0))*2+gap;al.nb[2]=al.nb[1]*n;al.nb[3]=al.nb[2]*hk;}else if(layout){al.nb[2]=k*2;al.nb[1]=k*hk*2+gap;al.nb[3]=al.nb[1]*n;}
 auto bl=dense(k,m,hq,batch,bt,gap);if(!op&&layout){bl.nb[2]=k*xb;bl.nb[1]=k*hq*xb+gap;bl.nb[3]=bl.nb[1]*m;}
 Tensor a(pd,al),b(pd,bl),y(pd,dense(n,m,hq,batch));
 for(int h=0;h<hk;++h)for(int j=0;j<n;++j)for(int z=0;z<k;++z){float v=sample(z+j*3+h*5)*.5f;if(pattern==1)v=0;if(pattern==2)v=(h==hk-1&&j==n-1&&z==k-1)?60000.f:sample(z+j+h)*.001f;if(pattern==3||pattern==6)v=(z%3-1)*0.00000011920928955078125f;if(pattern==4)v=(h%4+1)*sample(z+j)*7.0f;a.put(v,z,j,h);}
 for(int ba=0;ba<batch;++ba)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<k;++z){float v=sample(z+j*7+h*11+ba*13)*.5f;if(pattern==1)v=0;if(pattern==2)v=(h==hq-1&&j==m-1&&z==k-1)?30000.f:sample(z+j+h)*.0001f;if(pattern==3)v=(z%5-2)*0.00000011920928955078125f;if(pattern==4)v=(h%4+1)*sample(z+j)*3.0f;if(pattern==7&&((h==hq-2&&j==std::min(31,m-1)&&z==k-1)||(h==hq-1&&j==0&&z==0)))v=30000.f;if(pattern==6)v=(z%5-2)*1e-9f;if(pattern==5)v=(1+(z%16)*0.0009765625f)+(j%3-1)*0.00048828125f;b.put(v,z,j,h,ba);}
 a.upload();b.upload();y.upload();auto ar=a.raw,br=b.raw;Matmul4Layout mm{a.p,b.p,y.p,true};std::vector<float>ref((size_t)n*m*hq*batch);
 size_t idx=0;for(int ba=0;ba<batch;++ba)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<n;++z){double sum=0;for(int t=0;t<k;++t)sum+=double(a.get(t,z,h/(hq/hk)))*float((_Float16)b.get(t,j,h,ba));ref[idx++]=sum;}
 std::vector<float>old,old_updated,old_view;double olderror=0;int pass=1;
 for(const char*mode:{"legacy","auto","dma"}){
  a.raw=ar;b.raw=br;a.upload();b.upload();
  if(std::getenv("KUNLUN_PREP_AUDIT"))setenv("K200_PROFILE_PHASE",(std::string("warmup.")+mode).c_str(),1);
  setenv("KUNLUN_ATTENTION_PREP",mode,1);Kernel kernel;auto run=[&]{return attention_matmul(kernel,pd,y.ptr(),y.bytes(),a.ptr(),a.bytes(),b.ptr(),b.bytes(),mm);};require(run());if(std::getenv("KUNLUN_PREP_AUDIT"))setenv("K200_PROFILE_PHASE",mode,1);std::vector<double>times;
  for(int t=0;t<7;++t){auto start=std::chrono::steady_clock::now();require(run());times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
  y.download();Error cpu,delta;std::vector<float>value;int changed=0;idx=0;
  for(int ba=0;ba<batch;++ba)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<n;++z){float v=y.get(z,j,h,ba);value.push_back(v);cpu.add(v,ref[idx]);if(!old.empty()){delta.add(v,old[idx]);if(memcmp(&v,&old[idx],4))++changed;}++idx;}
  if(old.empty()){old=value;olderror=cpu.nmse();}a.download();b.download();int guards=y.guards()+(a.raw!=ar)+(b.raw!=br);std::sort(times.begin(),times.end());auto st=attention_statistics(kernel);int ok=!cpu.bad&&!delta.bad&&!guards&&delta.nmse()<1e-10&&cpu.nmse()<=std::max(2e-6,olderror*1.001);pass&=ok;
  printf("MATMUL_UNIT pd=%d op=%s D=%d S=%d M=%d HK=%d HQ=%d xb=%d pattern=%d gap=%d batch=%d layout=%d prep=%s ms=%.6f cpu_nmse=%.12g cpu_max=%.12g old_nmse=%.12g old_max=%.12g changed=%d guards=%d hbm=%zu pack=%zu pass=%d\n",pd,op?"KQV":"KQ",d,s,m,hk,hq,xb,pattern,gap,batch,layout,mode,times[3],cpu.nmse(),cpu.max,delta.nmse(),delta.max,changed,guards,st.hbm_workspace_bytes,st.pack_calls,ok);fflush(stdout);
  if(std::getenv("KUNLUN_PREP_MUTATE")){
   a.put(32.f,k-1,n-1,hk-1);b.put(-16.f,k-1,m-1,hq-1,batch-1);auto updated_a=a.raw,updated_b=b.raw;a.upload();b.upload();require(run());y.download();std::vector<float>updated;Error de;int changes=0;idx=0;
   for(int ba=0;ba<batch;++ba)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<n;++z){float v=y.get(z,j,h,ba);updated.push_back(v);if(!old_updated.empty()){de.add(v,old_updated[idx]);if(memcmp(&v,&old_updated[idx],4))++changes;}++idx;}
   if(old_updated.empty())old_updated=updated;require(!de.bad&&de.nmse()==0&&changes==0);
   auto view=mm;view.b.ne[1]=1;view.y=dense(n,1,hq,batch);Tensor one(pd,view.y);one.upload();require(attention_matmul(kernel,pd,one.ptr(),one.bytes(),a.ptr(),a.bytes(),b.ptr(),b.bytes(),view));one.download();std::vector<float>vv;idx=0;Error ve;
   for(int ba=0;ba<batch;++ba)for(int h=0;h<hq;++h)for(int z=0;z<n;++z){float v=one.get(z,0,h,ba);vv.push_back(v);if(!old_view.empty())ve.add(v,old_view[idx]);++idx;}
   if(old_view.empty())old_view=vv;require(!ve.bad&&ve.nmse()==0&&one.guards()==0);a.download();b.download();require(a.raw==updated_a&&b.raw==updated_b);
   printf("MUTATE_CHECK prep=%s pd=%d op=%s updated_nmse=%.12g changed=%d view_nmse=%.12g pass=1\n",mode,pd,op?"KQV":"KQ",de.nmse(),changes,ve.nmse());fflush(stdout);
  }

 }
 return pass?0:1;
}
