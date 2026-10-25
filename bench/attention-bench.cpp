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
template<class F>static void measure(const char *name,F run,Error e,int guards,int pd){
 require(run());std::vector<double>times;
 for(int i=0;i<7;++i){auto t=std::chrono::steady_clock::now();require(run());times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
 std::sort(times.begin(),times.end());bool pass=!e.bad && !guards && e.nmse()<((std::getenv("KUNLUN_ATTN_TEST_HALF") && (!std::strcmp(name,"KQ") || !std::strcmp(name,"KQV")))?2e-6:1e-9);
 std::printf("ATTENTION_UNIT op=%s pd=%d nmse=%.12g max_abs=%.12g bad=%d guards=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f pass=%d\n",name,pd,e.nmse(),e.max,e.bad,guards,times[3],times.front(),times.back(),pass);std::fflush(stdout);require(pass);
}
int main(int argc,char **argv){
 int pd=argc>1?std::atoi(argv[1]):0,d=argc>2?std::atoi(argv[2]):64,s=argc>3?std::atoi(argv[3]):257,m=argc>4?std::atoi(argv[4]):3;
 int hk=argc>5?std::atoi(argv[5]):4,hq=argc>6?std::atoi(argv[6]):32,bs=argc>7?std::atoi(argv[7]):2,past=argc>8?std::atoi(argv[8]):7,gap=argc>9?std::atoi(argv[9]):16,batch=argc>10?std::atoi(argv[10]):1;
 require(pd>=0 && pd<2 && d>0 && d<=512 && s>0 && s<=16384 && m>0 && m<=256 && hq%hk==0 && batch>0 && batch<=4 && (bs==2 || bs==4));
 int pattern=std::getenv("KUNLUN_ATTN_TEST_PATTERN")?std::atoi(std::getenv("KUNLUN_ATTN_TEST_PATTERN")):0;
 auto type=bs==2?InputType::f16:InputType::f32;
 Tensor4Layout kl={{d,s,hk,1},{bs,d*hk*bs+gap,d*bs,(d*hk*bs+gap)*(s+17)},type};
 Tensor4Layout ql={{d,m,hq,batch},{4,d*hq*4+gap,d*4,(d*hq*4+gap)*m},InputType::f32};
 Tensor k(pd,kl),q(pd,ql),score(pd,dense(s,m,hq,batch)),prob(pd,dense(s,m,hq,batch)),mask(pd,dense(s,m,pattern==6?2:1,1,type));
 for(int h=0;h<hk;++h)for(int j=0;j<s;++j)for(int z=0;z<d;++z)k.put(sample(z+j*3+h*5)*.5f,z,j,h);
 for(int b=0;b<batch;++b)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<d;++z)q.put(sample(z+j*7+h*11+b*13)*.5f,z,j,h,b);
 if(pattern==1 || pattern==2 || pattern==5) {
  for(int h=0;h<hk;++h)for(int j=0;j<s;++j)for(int z=0;z<d;++z)k.put(pattern==1?0:pattern==2?(z%17==0?60000.f:sample(z+j+h)):(z%3-1)*0.00000011920928955078125f,z,j,h);
  for(int b=0;b<batch;++b)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<d;++z)q.put(pattern==1?0:pattern==2?sample(z+j+h)*2:(z%5-2)*0.00000011920928955078125f,z,j,h,b);
 }
 k.upload();q.upload();score.upload();prob.upload();
 Kernel kernel;Matmul4Layout mm{k.p,q.p,score.p,true};
 require(attention_matmul(kernel,pd,score.ptr(),score.bytes(),k.ptr(),k.bytes(),q.ptr(),q.bytes(),mm));score.download();Error ke;
 for(int b=0;b<batch;++b)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<s;++z){double ref=0;for(int t=0;t<d;++t){float bv=q.get(t,j,h,b);if(bs==2)bv=(float)(_Float16)bv;ref+=double(k.get(t,z,h/(hq/hk)))*bv;}if(std::getenv("KUNLUN_ATTN_TEST_DEBUG") && std::abs(score.get(z,j,h,b)-ref)>.01){static int shown=0;if(shown++<24)std::printf("MISMATCH h=%d m=%d n=%d actual=%g ref=%g\n",h,j,z,score.get(z,j,h,b),ref);}ke.add(score.get(z,j,h,b),ref);}
 measure("KQ",[&]{return attention_matmul(kernel,pd,score.ptr(),score.bytes(),k.ptr(),k.bytes(),q.ptr(),q.bytes(),mm);},ke,score.guards(),pd);
 for(int h=0;h<mask.p.ne[2];++h)for(int j=0;j<m;++j)for(int z=0;z<s;++z)mask.put((pattern==3 && j==0)?-INFINITY:z<=past+j?(z%5)*-.125f*(h+1):-INFINITY,z,j,h);
 mask.upload();Softmax4Layout sm{score.p,prob.p,mask.p,true,1/std::sqrt((float)d),pattern==4?8.f:0.f};
 require(attention_softmax(kernel,pd,prob.ptr(),prob.bytes(),score.ptr(),score.bytes(),mask.ptr(),mask.bytes(),sm));prob.download();Error se;
 std::vector<float>expected((size_t)s*m*hq*batch);
 for(int b=0;b<batch;++b)for(int h=0;h<hq;++h)for(int j=0;j<m;++j){float slope=1;if(sm.max_bias>0){int nh=1;while(nh*2<=hq)nh*=2;float b0=std::pow(2.f,-sm.max_bias/nh),b1=std::pow(2.f,-sm.max_bias/(2*nh));slope=h<nh?std::pow(b0,(float)(h+1)):std::pow(b1,(float)(2*(h-nh)+1));}double mx=-INFINITY,sum=0;for(int z=0;z<s;++z)mx=std::max(mx,double(score.get(z,j,h,b))*sm.scale+slope*mask.get(z,j,h%mask.p.ne[2]));if(mx==-INFINITY){for(int z=0;z<s;++z){expected[((size_t)b*hq*m+h*m+j)*s+z]=0;se.add(prob.get(z,j,h,b),0);}continue;}for(int z=0;z<s;++z)sum+=std::exp(double(score.get(z,j,h,b))*sm.scale+slope*mask.get(z,j,h%mask.p.ne[2])-mx);for(int z=0;z<s;++z){float v=std::exp(double(score.get(z,j,h,b))*sm.scale+slope*mask.get(z,j,h%mask.p.ne[2])-mx)/sum;expected[((size_t)b*hq*m+h*m+j)*s+z]=v;se.add(prob.get(z,j,h,b),v);}}
 measure("Softmax",[&]{return attention_softmax(kernel,pd,prob.ptr(),prob.bytes(),score.ptr(),score.bytes(),mask.ptr(),mask.bytes(),sm);},se,prob.guards(),pd);
 auto vl=dense(s,d,hk,1,type,gap);vl.nb[1]=(s+17)*bs+gap;vl.nb[2]=vl.nb[1]*d;vl.nb[3]=vl.nb[2]*hk;
 Tensor v(pd,vl),out(pd,dense(d,m,hq,batch));
 for(int h=0;h<hk;++h)for(int j=0;j<d;++j)for(int z=0;z<s;++z)v.put(sample(z+j*3+h*7),z,j,h);
 v.upload();out.upload();mm={v.p,prob.p,out.p,true};
 require(attention_matmul(kernel,pd,out.ptr(),out.bytes(),v.ptr(),v.bytes(),prob.ptr(),prob.bytes(),mm));out.download();Error ve,ae;
 for(int b=0;b<batch;++b)for(int h=0;h<hq;++h)for(int j=0;j<m;++j)for(int z=0;z<d;++z){double ref=0,attn=0;for(int t=0;t<s;++t){float p=prob.get(t,j,h,b),p0=expected[((size_t)b*hq*m+h*m+j)*s+t];if(bs==2){p=(float)(_Float16)p;p0=(float)(_Float16)p0;}ref+=double(v.get(t,z,h/(hq/hk)))*p;attn+=double(v.get(t,z,h/(hq/hk)))*p0;}ve.add(out.get(z,j,h,b),ref);ae.add(out.get(z,j,h,b),attn);}
 measure("KQV",[&]{return attention_matmul(kernel,pd,out.ptr(),out.bytes(),v.ptr(),v.bytes(),prob.ptr(),prob.bytes(),mm);},ve,out.guards(),pd);
 std::printf("ATTENTION_CHAIN pd=%d nmse=%.12g max_abs=%.12g\n",pd,ae.nmse(),ae.max);require(!ae.bad && ae.nmse()<1e-6);
 // Permuted KQV output is flattened by CONT before the output projection.
 auto cp=out.p;std::swap(cp.ne[1],cp.ne[2]);std::swap(cp.nb[1],cp.nb[2]);Tensor flattened(pd,dense(d*hq,m,1,batch));flattened.upload();
 require(attention_contiguous(kernel,pd,flattened.ptr(),flattened.bytes(),out.ptr(),out.bytes(),cp));flattened.download();Error ce;
 for(int b=0;b<batch;++b)for(int j=0;j<m;++j)for(int h=0;h<hq;++h)for(int z=0;z<d;++z)ce.add(flattened.get(h*d+z,j,0,b),out.get(z,j,h,b));
 measure("CONT",[&]{return attention_contiguous(kernel,pd,flattened.ptr(),flattened.bytes(),out.ptr(),out.bytes(),cp);},ce,flattened.guards(),pd);
 if(d%2==0){
  Tensor rx(pd,dense(d,hq,m,1,type,gap)),ry(pd,dense(d,hq,m,1,type));std::vector<int32_t>pos(m);for(int j=0;j<m;++j)pos[j]=past+j;
  Buffer positions(pd,m*4);check(copy(positions.data(),pos.data(),m*4,host_to_device));check(wait());
  for(int j=0;j<m;++j)for(int h=0;h<hq;++h)for(int z=0;z<d;++z)rx.put(sample(z+j*5+h*7),z,h,j);rx.upload();ry.upload();
  bool neox=std::getenv("KUNLUN_ATTN_TEST_NEOX");Rope4Layout rp{rx.p,ry.p,neox?d/2:d,neox?2:0,neox?2:0,10000,neox?.75f:1.f,neox?1.1f:1.f};require(attention_rope(kernel,pd,ry.ptr(),ry.bytes(),rx.ptr(),rx.bytes(),(int32_t*)positions.data(),m*4,rp));ry.download();Error re;float theta_scale=std::pow(rp.freq_base,-2.f/rp.n_dims);
  for(int j=0;j<m;++j)for(int h=0;h<hq;++h){std::vector<float>ref(d);for(int z=0;z<d;++z)ref[z]=rx.get(z,h,j);float theta=pos[j];for(int z=0;z<rp.n_dims/2;++z){int i=rp.n_offs+(rp.mode==0?2*z:z),k=i+(rp.mode==0?1:rp.n_dims/2);float sn=std::sin(theta*rp.freq_scale)*rp.attn_factor,cs=std::cos(theta*rp.freq_scale)*rp.attn_factor,a=rx.get(i,h,j),b=rx.get(k,h,j);ref[i]=a*cs-b*sn;ref[k]=a*sn+b*cs;theta*=theta_scale;}for(int z=0;z<d;++z){float refv=bs==2?(float)(_Float16)ref[z]:ref[z];re.add(ry.get(z,h,j),refv);}}

  measure("RoPE",[&]{return attention_rope(kernel,pd,ry.ptr(),ry.bytes(),rx.ptr(),rx.bytes(),(int32_t*)positions.data(),m*4,rp);},re,ry.guards(),pd);
 }
 auto original_k=k.raw,original_v=v.raw;
 k.download();v.download();require(k.raw==original_k && v.raw==original_v);
 std::printf("ATTENTION_READ_ONLY_KV pd=%d PASS\n",pd);
 auto stats=attention_statistics(kernel);std::printf("ATTENTION_WORKSPACE pd=%d hbm=%zu l3=%zu host_trig=%zu pack=%zu\n",pd,stats.hbm_workspace_bytes,stats.l3_workspace_bytes,stats.rope_host_trig,stats.pack_calls);
}
