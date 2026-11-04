#include <kunlun/k200.h>
#include "xpu/runtime.h"
#include "native-q4-reference.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
static void ensure(bool v,const char *s){if(!v)throw std::runtime_error(s);}
static void put(void*d,const void*s,size_t n){kunlun::check(kunlun::copy(d,s,n,kunlun::host_to_device));kunlun::check(kunlun::wait());}
static std::vector<unsigned char> read(const void*p,size_t n){std::vector<unsigned char>b(n);kunlun::check(kunlun::copy(b.data(),p,n,kunlun::device_to_host));kunlun::check(kunlun::wait());return b;}
struct Arg{const void*p;size_t n;};template<class T>static Arg arg(const T&v){return{&v,sizeof v};}
struct Program{
 kunlun::Buffer code;XPUFunc fn{};int cores;
 static std::vector<char>load(const std::string&p){std::ifstream f(p,std::ios::binary);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});ensure(!b.empty(),"image missing");b.resize(((b.size()+255)&~size_t(255))+256);ensure(b.size()<=16384,"code bound");return b;}
 Program(int pd,const std::string&base,const std::string&name,bool sd):Program(pd,load(base+"/"+name+".bin"),base+"/"+name+".entry",sd){}
 Program(int pd,const std::vector<char>&b,const std::string&e,bool sd):code(pd,b.size()),cores(sd?8:16){std::ifstream f(e);std::string symbol;std::getline(f,symbol);ensure(!symbol.empty(),"entry");put(code.data(),b.data(),b.size());if(sd)kunlun::check(xpu_create_sd_func(&fn,(uint64_t)(uintptr_t)code.data(),b.size(),0,0,0,symbol.c_str(),true));else kunlun::check(xpu_create_cl_func(&fn,(uint64_t)(uintptr_t)code.data(),b.size(),0,0,0,symbol.c_str(),true));}
 void run(std::initializer_list<Arg>a){size_t off=0;for(auto v:a){kunlun::check(xpu_launch_argument_set(v.p,v.n,off));off+=v.n;}kunlun::check(xpu_launch_config(4,cores));kunlun::check(xpu_launch_async(fn));kunlun::check(xpu_wait());}
};
struct L3{void*p=nullptr;size_t bytes;L3(size_t n):bytes(n){kunlun::check(xpu_malloc(&p,n,XPU_MEM_L3));ensure((uintptr_t)p%256==0,"L3 alignment");}~L3(){kunlun::check(xpu_wait());kunlun::check(xpu_free(p));}};
static std::vector<unsigned char>weights(int k,int n,uint32_t seed){int blocks=k/32;std::vector<unsigned char>b((size_t)n*blocks*18);for(int row=0;row<n;++row)for(int block=0;block<blocks;++block){auto*p=b.data()+((size_t)row*blocks+block)*18;auto h=native_q4_bench::random(seed);float d=std::ldexp((8+h%9)/1024.f,int((h>>4)%7)-3);if(h&0x80000000)d=-d;_Float16 half=(_Float16)d;std::memcpy(p,&half,2);for(int j=0;j<16;++j)p[j+2]=native_q4_bench::random(seed);}return b;}
static void samples(const char*key,const std::vector<double>&v){std::printf(",\"%s\":[",key);for(size_t i=0;i<v.size();++i)std::printf("%s%.9f",i?",":"",v[i]);std::printf("]");}
int main(int argc,char**argv){try{
 ensure(argc==6,"PD K N pattern trials");int pd=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]),pattern=std::stoi(argv[4]),trials=std::stoi(argv[5]);
 ensure(pd>=0&&pd<=1&&((k==4096&&n==11008)||(k==2048&&n==5632))&&pattern>=0&&pattern<=3&&trials>=3&&trials<=51,"arguments");
 std::lock_guard<std::recursive_mutex>lock(kunlun::runtime_mutex());kunlun::select(pd);
 auto raw0=weights(k,n,0x12345678),raw1=weights(k,n,0x87654321);kunlun::CommonQ4Weight w0(pd,k,n),w1(pd,k,n);w0.upload(raw0.data(),raw0.size());w1.upload(raw1.data(),raw1.size());
 auto weight0=w0.data(),weight1=w1.data();const void*scale0=(const char*)weight0+w0.info().scale_offset,*scale1=(const char*)weight1+w1.info().scale_offset;
 std::string olddir=std::getenv("KUNLUN_KERNEL_DIR"),newdir=std::getenv("K200_PAIR_PROBE_DIR"),route=k==4096?"nmajor":"stream";
 Program oldsd(pd,olddir,"kunlun-common-gemv-addr32-"+route,true),oldsum(pd,olddir,"kunlun-common-sum",false),pairsd(pd,newdir,"kunlun-common-gemv-pair-"+route,true),pairsum(pd,newdir,"kunlun-common-sum-pair",false);
 const size_t partial_bytes=(size_t)n*(k/32)*4;L3 l3(partial_bytes*2+1024);void*p0=(char*)l3.p+256,*p1=(char*)l3.p+partial_bytes+768;
 std::vector<unsigned char>pguard(l3.bytes,0xa5);put(l3.p,pguard.data(),pguard.size());
 kunlun::Buffer input(pd,k*4+512),out0(pd,n*4+512),out1(pd,n*4+512);void*x=(char*)input.data()+256,*y0=(char*)out0.data()+256,*y1=(char*)out1.data()+256;
 kunlun::CommonQ4Workspace workspace;kunlun::CommonQ4ActivationLease lease;uint64_t generation=0;
 int batch=1,first=0,components=2,bits=8,pipelined=1,fold=0,mode=1;float zero=0,one=1;
 const void*q=nullptr,*params=nullptr;
 auto prepare=[&](){ensure(kunlun::common_q4_prepare(workspace,pd,x,k,1,kunlun::InputType::f32,++generation,lease),"prepare");auto c=lease.chunk(0);ensure(c.components==2,"components");q=c.values;params=c.scales;};
 auto sd=[&](void*p,const void*w,const void*s){oldsd.run({arg(p),arg(w),arg(q),arg(k),arg(n),arg(batch),arg(first),arg(n),arg(components),arg(bits),arg(zero),arg(zero),arg(one),arg(pipelined),arg(s),arg(fold),arg(params)});};
 auto sum=[&](void*y,void*p){oldsum.run({arg(y),arg(p),arg(scale0),arg(k),arg(n),arg(batch),arg(first),arg(n),arg(mode)});};
 auto dualsd=[&](){pairsd.run({arg(p0),arg(weight0),arg(q),arg(k),arg(n),arg(batch),arg(first),arg(n),arg(components),arg(bits),arg(zero),arg(zero),arg(one),arg(pipelined),arg(scale0),arg(fold),arg(params),arg(weight1),arg(scale1),arg(p1)});};
 auto dualsum=[&](){pairsum.run({arg(y0),arg(p0),arg(scale0),arg(k),arg(n),arg(batch),arg(first),arg(n),arg(mode),arg(y1),arg(p1)});};
 std::vector<double>times[3][4];
 auto pipeline=[&](int variant,bool prof){auto start=Clock::now();prepare();auto t1=prof?Clock::now():start;double sdt=0,sumt=0;
  if(variant==0||variant==2){auto a=prof?Clock::now():start;sd(p0,weight0,scale0);auto b=prof?Clock::now():start;sum(y0,p0);auto c=prof?Clock::now():start;void*target=variant==2?p1:p0;sd(target,weight1,scale1);auto d=prof?Clock::now():start;sum(y1,target);if(prof){auto e=Clock::now();sdt=std::chrono::duration<double,std::milli>(b-a+d-c).count();sumt=std::chrono::duration<double,std::milli>(c-b+e-d).count();}}
  else{dualsd();auto b=prof?Clock::now():start;dualsum();if(prof){auto c=Clock::now();sdt=std::chrono::duration<double,std::milli>(b-t1).count();sumt=std::chrono::duration<double,std::milli>(c-b).count();}}
  auto end=Clock::now();return std::vector<double>{std::chrono::duration<double,std::milli>(end-start).count(),prof?std::chrono::duration<double,std::milli>(t1-start).count():0,sdt,sumt};};
 for(int epoch=0;epoch<2;++epoch){std::vector<float>values(k);for(int j=0;j<k;++j){float v=3.3f*std::sin(j*.071f+epoch*.13f);if(pattern==1)v=0;if(pattern==2)v=-std::fabs(v);if(pattern==3)v=j%4==0?1e30f:j%4==1?-1e30f:j%4==2?1e-35f:-1e-35f;values[j]=v;}
  std::vector<unsigned char>ix(input.size(),0x91),ox(out0.size(),0x7a);std::memcpy(ix.data()+256,values.data(),k*4);put(input.data(),ix.data(),ix.size());put(out0.data(),ox.data(),ox.size());put(out1.data(),ox.data(),ox.size());
  prepare();auto qgold=read(q,k*4+1024),sgold=read(params,64);sd(p0,weight0,scale0);auto pgold0=read(p0,partial_bytes);sum(y0,p0);sd(p0,weight1,scale1);auto pgold1=read(p0,partial_bytes);sum(y1,p0);auto gold0=read(out0.data(),out0.size()),gold1=read(out1.data(),out1.size());
  put(l3.p,pguard.data(),pguard.size());put(out0.data(),ox.data(),ox.size());put(out1.data(),ox.data(),ox.size());pipeline(1,false);
  ensure(read(q,k*4+1024)==qgold&&read(params,64)==sgold,"quantization");ensure(read(p0,partial_bytes)==pgold0&&read(p1,partial_bytes)==pgold1,"partial bitwise");ensure(read(out0.data(),out0.size())==gold0&&read(out1.data(),out1.size())==gold1,"output bitwise");ensure(read(input.data(),input.size())==ix,"input modified");
  auto whole=read(l3.p,l3.bytes);for(size_t j=0;j<256;++j)ensure(whole[j]==0xa5&&whole[partial_bytes+256+j]==0xa5&&whole[partial_bytes+512+j]==0xa5&&whole[l3.bytes-1-j]==0xa5,"L3 guard");
 }
 for(int warm=0;warm<3;++warm)for(int variant=0;variant<3;++variant)pipeline(variant,false);
 for(int trial=0;trial<trials;++trial)for(int step=0;step<2;++step){int variant=trial%2?1-step:step;auto t=pipeline(variant,false);times[variant][0].push_back(t[0]);}
 std::vector<double>profile[3][4];for(int trial=0;trial<7;++trial)for(int step=0;step<3;++step){int variant=trial%2?2-step:step;auto t=pipeline(variant,true);for(int j=0;j<4;++j)profile[variant][j].push_back(t[j]);}
 std::printf("PAIR_RESULT {\"pd\":%d,\"k\":%d,\"n\":%d,\"pattern\":%d,\"bitwise\":true,\"guard\":true,\"partial_bytes_each\":%zu,\"l3_bytes\":%zu",pd,k,n,pattern,partial_bytes,l3.bytes);
 samples("old_total_ms",times[0][0]);samples("pair_total_ms",times[1][0]);const char*fields[]={"total","quant","sdnn","sum"};for(int variant=0;variant<3;++variant)for(int j=0;j<4;++j){auto key=std::string(variant==0?"old_":variant==1?"pair_":"old_split_")+fields[j]+"_profile_ms";samples(key.c_str(),profile[variant][j]);}std::puts("}");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"PAIR_FAIL %s\n",e.what());return 1;}}
