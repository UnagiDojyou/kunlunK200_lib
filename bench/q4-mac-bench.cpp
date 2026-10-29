#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using namespace native_q4_bench;
using Clock=std::chrono::steady_clock;
static void ensure(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
static std::vector<unsigned char> download(const void* source,size_t bytes){
    std::vector<unsigned char> result(bytes);kunlun::check(kunlun::copy(result.data(),source,bytes,kunlun::device_to_host));kunlun::check(kunlun::wait());return result;
}
static void samples(const char* key,const std::vector<double>& values){
    std::printf("\"%s\":[",key);for(size_t i=0;i<values.size();++i)std::printf("%s%.9f",i?",":"",values[i]);std::printf("]");
}
int main(int argc,char**argv){try{
    ensure(argc==8,"k200-q4-mac-bench PD K N M f32|f16 pattern trials");
    int pd=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]),m=std::stoi(argv[4]),pattern=std::stoi(argv[6]),trials=std::stoi(argv[7]);
    bool f16=std::string(argv[5])=="f16";ensure(pd>=0&&pd<2&&m>0&&m<=1024&&trials>=3&&trials<=31&&kunlun::common_q4_supported(k,n),"shape");
    std::lock_guard<std::recursive_mutex> lock(kunlun::runtime_mutex());kunlun::select(pd);
    int blocks=(k+31)/32;std::vector<unsigned char> raw((size_t)n*blocks*18);uint32_t rng=0x7340d123;
    for(int row=0;row<n;++row)for(int block=0;block<blocks;++block){
        auto p=raw.data()+((size_t)row*blocks+block)*18;uint32_t h=random(rng);float scale=std::ldexp((8+h%9)/1024.f,int((h>>4)%7)-3);
        if(h&0x80000000)scale=-scale;if((row*13+block)%257==0)scale=0;_Float16 fp16=(_Float16)scale;std::memcpy(p,&fp16,2);for(int j=0;j<16;++j)p[2+j]=random(rng);
    }
    auto x=input(k,m,pattern);std::vector<_Float16> hx(x.size());if(f16)for(size_t i=0;i<x.size();++i)hx[i]=(_Float16)x[i];
    auto type=f16?kunlun::InputType::f16:kunlun::InputType::f32;
    kunlun::Buffer dx(pd,x.size()*(f16?2:4)),dy(pd,(size_t)m*n*4+512);
    kunlun::check(kunlun::copy(dx.data(),f16?(const void*)hx.data():x.data(),dx.size(),kunlun::host_to_device));
    std::vector<unsigned char> poison(dy.size(),0xa7);kunlun::check(kunlun::copy(dy.data(),poison.data(),poison.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
    void* y=(char*)dy.data()+256;kunlun::CommonQ4Weight weight(pd,k,n);weight.upload(raw.data(),raw.size());auto info=weight.info();auto weight_before=download(weight.data(),info.payload_bytes);
    kunlun::CommonQ4Workspace ordinary,prepared;kunlun::CommonQ4ActivationLease lease;
    setenv("KUNLUN_Q4_MAC_DMA","0",1);kunlun::common_q4_matmul(ordinary,pd,y,weight,dx.data(),m,type);auto gold=download(dy.data(),dy.size());
    bool available=kunlun::common_q4_prepare(prepared,pd,dx.data(),k,m,type,7,lease);
    std::vector<std::vector<unsigned char>> quant;
    if(available)for(size_t i=0;i<lease.chunks();++i){auto c=lease.chunk(i);quant.push_back(download(c.values,m==1?4*info.kp:(size_t)c.padded_rows*info.kp));quant.push_back(download(c.scales,m==1?64:(size_t)c.padded_rows*64));}
    auto invoke=[&](int mode){
        setenv("KUNLUN_Q4_MAC_DMA",mode&1?"1":"0",1);
        if(mode<2)kunlun::common_q4_matmul(ordinary,pd,y,weight,dx.data(),m,type);
        else ensure(kunlun::common_q4_matmul_shared(prepared,pd,y,weight,lease,7),"prepared execution refused");
    };
    int modes=available?4:2;std::vector<double> times[4];
    for(int mode=0;mode<modes;++mode){for(int i=0;i<3;++i)invoke(mode);ensure(download(dy.data(),dy.size())==gold,"warmup bitwise mismatch");}
    for(int trial=0;trial<trials;++trial)for(int step=0;step<modes;++step){
        int mode=trial%2?modes-1-step:step;
        setenv("KUNLUN_Q4_MAC_DMA",mode&1?"1":"0",1);
        auto begin=Clock::now();
        if(mode<2)kunlun::common_q4_matmul(ordinary,pd,y,weight,dx.data(),m,type);
        else ensure(kunlun::common_q4_matmul_shared(prepared,pd,y,weight,lease,7),"prepared execution refused");
        times[mode].push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
        ensure(download(dy.data(),dy.size())==gold,"output bitwise mismatch");
    }
    if(available)for(size_t i=0;i<lease.chunks();++i){auto c=lease.chunk(i);ensure(download(c.values,m==1?4*info.kp:(size_t)c.padded_rows*info.kp)==quant[2*i],"quant changed");ensure(download(c.scales,m==1?64:(size_t)c.padded_rows*64)==quant[2*i+1],"activation scale changed");}
    ensure(download(weight.data(),info.payload_bytes)==weight_before,"weight changed");ensure(weight.statistics().uploads==1,"reupload");
    for(size_t i=0;i<256;++i)ensure(gold[i]==0xa7&&gold[gold.size()-1-i]==0xa7,"output guard");
    auto a=ordinary.mac_statistics(),b=prepared.mac_statistics();auto aw=ordinary.statistics(),bw=prepared.statistics();
    // Invalid modes retain the stable path.
    setenv("KUNLUN_Q4_MAC_DMA","invalid-test",1);kunlun::common_q4_matmul(ordinary,pd,y,weight,dx.data(),m,type);ensure(download(dy.data(),dy.size())==gold,"invalid mode fallback");
    std::printf("MAC_API_RESULT {\"pd\":%d,\"k\":%d,\"n\":%d,\"m\":%d,\"f16\":%s,\"pattern\":%d,\"prepared\":%s,\"bitwise\":true,\"quant_unchanged\":true,\"weight_unchanged\":true,\"uploads\":1,\"baseline_sd\":%zu,\"stream_sd\":%zu,\"shape_fallback\":%zu,\"resident_l1d\":%zu,\"ordinary_hbm\":%zu,\"prepared_hbm\":%zu,\"l3\":%zu,",pd,k,n,m,f16?"true":"false",pattern,available?"true":"false",a.baseline_sd_calls+b.baseline_sd_calls,a.stream_sd_calls+b.stream_sd_calls,a.shape_fallbacks+b.shape_fallbacks,std::max(a.resident_l1d_bytes,b.resident_l1d_bytes),aw.hbm_workspace_bytes,bw.hbm_workspace_bytes,std::max(aw.l3_workspace_bytes,bw.l3_workspace_bytes));
    samples("ordinary0_ms",times[0]);std::printf(",");samples("ordinary1_ms",times[1]);std::printf(",");samples("prepared0_ms",times[2]);std::printf(",");samples("prepared1_ms",times[3]);std::puts("}");
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"MAC_API_FAIL %s\n",e.what());return 1;}}
