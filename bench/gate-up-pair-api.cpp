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
 kunlun::Buffer input(pd,k*4+512),out0(pd,n*4+512),out1(pd,n*4+512);void*x=(char*)input.data()+256,*y0=(char*)out0.data()+256,*y1=(char*)out1.data()+256;
 kunlun::CommonQ4Workspace workspace;kunlun::CommonQ4ActivationLease lease;uint64_t generation=0;
 auto prepare=[&](){ensure(kunlun::common_q4_prepare(workspace,pd,x,k,1,kunlun::InputType::f32,++generation,lease),"prepare");};
 auto pair=[&](){return kunlun::common_q4_matmul_pair(workspace,pd,y0,y1,w0,w1,lease,generation);};
 auto invoke=[&](int mode){setenv("GGML_KUNLUN_Q4_GATE_UP_PAIR",mode?"1":"0",1);auto start=Clock::now();prepare();
  if(mode)ensure(pair(),"pair API refused");else{ensure(kunlun::common_q4_matmul_shared(workspace,pd,y0,w0,lease,generation),"old gate");ensure(kunlun::common_q4_matmul_shared(workspace,pd,y1,w1,lease,generation),"old up");}
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
 for(int epoch=0;epoch<2;++epoch){std::vector<float>values(k);for(int j=0;j<k;++j){float v=3.3f*std::sin(j*.071f+epoch*.13f);if(pattern==1)v=0;if(pattern==2)v=-std::fabs(v);if(pattern==3)v=j%4==0?1e30f:j%4==1?-1e30f:j%4==2?1e-35f:-1e-35f;values[j]=v;}
  std::vector<unsigned char>ix(input.size(),0x91),ox(out0.size(),0x7a);std::memcpy(ix.data()+256,values.data(),k*4);put(input.data(),ix.data(),ix.size());put(out0.data(),ox.data(),ox.size());put(out1.data(),ox.data(),ox.size());
  invoke(0);auto c=lease.chunk(0);auto qgold=read(c.values,k*4+1024),sgold=read(c.scales,64),gold0=read(out0.data(),out0.size()),gold1=read(out1.data(),out1.size());
  put(out0.data(),ox.data(),ox.size());put(out1.data(),ox.data(),ox.size());invoke(1);c=lease.chunk(0);
  ensure(read(c.values,k*4+1024)==qgold&&read(c.scales,64)==sgold,"quant bits");ensure(read(out0.data(),out0.size())==gold0&&read(out1.data(),out1.size())==gold1,"output bits/guards");
  ensure(read(input.data(),input.size())==ix,"input guard");
  ensure(kunlun::common_q4_matmul_shared(workspace,pd,y0,w0,lease,generation),"lease after pair");ensure(read(out0.data(),out0.size())==gold0,"third consumer bits");
 }
 for(int i=0;i<3;++i){invoke(0);invoke(1);}std::vector<double>times[2];
 for(int trial=0;trial<trials;++trial)for(int step=0;step<2;++step){int mode=trial%2?1-step:step;times[mode].push_back(invoke(mode));}
 auto before0=read(out0.data(),out0.size()),before1=read(out1.data(),out1.size());
 for(const char *flag:{"0","invalid"}){setenv("GGML_KUNLUN_Q4_GATE_UP_PAIR",flag,1);ensure(!pair(),"disabled API accepted");}
 unsetenv("GGML_KUNLUN_Q4_GATE_UP_PAIR");ensure(!pair(),"unset API accepted");setenv("GGML_KUNLUN_Q4_GATE_UP_PAIR","1",1);
 ensure(!kunlun::common_q4_matmul_pair(workspace,1-pd,y0,y1,w0,w1,lease,generation),"wrong PD accepted");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,y0,y1,w0,w1,lease,generation-1),"wrong generation");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,y0,y0,w0,w1,lease,generation),"output alias");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,x,y1,w0,w1,lease,generation),"input alias");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,(char*)y0+1,y1,w0,w1,lease,generation),"misalignment");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,(char*)y0+4,y1,w0,w1,lease,generation),"DMA misalignment");
 ensure(!kunlun::common_q4_matmul_pair(workspace,pd,y0,y1,w0,w0,lease,generation),"weight alias");
 {kunlun::CommonQ4Workspace other;ensure(!kunlun::common_q4_matmul_pair(other,pd,y0,y1,w0,w1,lease,generation),"wrong workspace");}
 kunlun::common_q4_matmul(workspace,pd,y0,w0,x,1);ensure(!pair(),"stale lease");
 ensure(read(out0.data(),out0.size())==before0&&read(out1.data(),out1.size())==before1,"fallback changed output");
 std::vector<_Float16>half(k,(_Float16).5f);kunlun::Buffer hx(pd,k*2);put(hx.data(),half.data(),k*2);
 ensure(kunlun::common_q4_prepare(workspace,pd,hx.data(),k,1,kunlun::InputType::f16,++generation,lease),"F16 prepare");ensure(!pair(),"F16 accepted");
 std::vector<float>two(k*2,.25f);kunlun::Buffer mx(pd,k*8);put(mx.data(),two.data(),k*8);
 ensure(kunlun::common_q4_prepare(workspace,pd,mx.data(),k,2,kunlun::InputType::f32,++generation,lease),"M2 prepare");ensure(!pair(),"M2 accepted");
 invoke(1);auto stats=kunlun::common_q4_pair_statistics(workspace);
 std::printf("PAIR_API_RESULT {\"pd\":%d,\"k\":%d,\"n\":%d,\"pattern\":%d,\"bitwise\":true,\"guards\":true,\"fallback\":true,\"calls\":%zu,\"declined\":%zu,\"l3_bytes\":%zu",pd,k,n,pattern,stats.calls,stats.declined,stats.l3_partial_bytes);samples("old_ms",times[0]);samples("pair_ms",times[1]);std::puts("}");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"PAIR_API_FAIL %s\n",e.what());return 1;}}
