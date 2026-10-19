#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <chrono>
#include <numeric>
#include <string>
#include <dlfcn.h>
using namespace native_q4_bench;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
int main(int argc,char **argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if (argc < 9) {
        std::fprintf(stderr,"usage: k200-native-q4-bench native|direct|cdnn|cpu PD K N batch f32|f16 pattern trials\n"); return 2;
    }
    std::string mode=argv[1],dtype=argv[6]; int pd=std::atoi(argv[2]),k=std::atoi(argv[3]),n=std::atoi(argv[4]),batch=std::atoi(argv[5]),pattern=std::atoi(argv[7]),trials=std::atoi(argv[8]);
    require(mode=="native" || mode=="direct" || mode=="cdnn" || mode=="cpu");
    require(kunlun::native_q4_layout_supported(k,n) && batch>=1 && batch<=32 && trials>=1 && trials<=50 && (dtype=="f32" || dtype=="f16"));
    auto begin=Clock::now(); Weights w(k,n); auto x=input(k,batch,pattern); std::vector<unsigned short> hx(x.size());
    if (dtype=="f16") for (size_t j=0;j<x.size();++j) { _Float16 h=(_Float16)x[j];std::memcpy(&hx[j],&h,2);x[j]=(float)h; }
    auto t=Clock::now(); auto gold=reference(w.decoded,x,k,n,batch); double cpu_ms=ms(t),fixture_ms=ms(begin);
    if (mode=="cpu") {
        std::vector<double> times;
        for(int trial=-1;trial<trials;++trial){t=Clock::now();auto result=reference(w.decoded,x,k,n,batch);require(result.size()==gold.size());double elapsed=ms(t);if(trial>=0){times.push_back(elapsed);std::printf("TRIAL index=%d ms=%.9f\n",trial,elapsed);}}
        std::sort(times.begin(),times.end());double mean=std::accumulate(times.begin(),times.end(),0.)/times.size(),var=0;
        for(auto v:times)var+=(v-mean)*(v-mean);
        std::printf("RESULT mode=cpu pd=%d k=%d n=%d batch=%d dtype=%s trials=%d median_ms=%.9f min_ms=%.9f max_ms=%.9f stddev_ms=%.9f weight_bytes=%zu\n",pd,k,n,batch,dtype.c_str(),trials,times[times.size()/2],times.front(),times.back(),std::sqrt(var/times.size()),w.decoded.size()*4);return 0;
    }
    setenv("K200_PROFILE_PHASE","setup",1);
    kunlun::select(pd); kunlun::check(kunlun::wait()); size_t before=kunlun::allocated_bytes(pd);
    kunlun::Buffer dx(pd,x.size()*(mode=="native" && dtype=="f16"?2:4)),dy(pd,(size_t)n*batch*4);
    kunlun::check(kunlun::copy(dx.data(),mode=="native" && dtype=="f16"?(void *)hx.data():(void *)x.data(),dx.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
    t=Clock::now(); size_t weights=mode=="native"?kunlun::native_q4_size(k,n):mode=="cdnn"?kunlun::packed_i8_size(k,n):w.q4.size();
    kunlun::Buffer dw(pd,weights);
    double packing_ms=0,layout_check_ms=0,upload_ms=0;
    if (mode=="native") {
        auto pack_start=Clock::now();
        std::vector<unsigned char> packed(weights);
        require(kunlun::pack_native_q4(packed.data(),packed.size(),w.q4.data(),k,n));
        packing_ms=ms(pack_start);auto check_start=Clock::now();
        // Verify every original nibble and FP16 scale through the inverse mapping.
        size_t payload=kunlun::aligned((((size_t)n+31)&~size_t(31))*k/2);
        for(int r=0;r<n;++r) for(int b=0;b<k/32;++b) {
            size_t tb=(size_t)(r/32)*(k/32)+b; auto *p=w.q4.data()+((size_t)r*(k/32)+b)*18;
            require(std::memcmp(p,packed.data()+payload+(tb*32+r%32)*2,2)==0);
            for(int j=0;j<32;++j){unsigned int v=packed[tb*512+(j/2)*32+r%32];require(signed4(v>>(j%2*4))==nibble(p,j));}
        }
        layout_check_ms=ms(check_start);auto upload_start=Clock::now();
        kunlun::check(kunlun::copy(dw.data(),packed.data(),packed.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());upload_ms=ms(upload_start);
    } else if(mode=="cdnn") kunlun::upload_i8_rows(pd,dw.data(),k,n,0,n,w.decoded.data());
    else { kunlun::check(kunlun::copy(dw.data(),w.q4.data(),w.q4.size(),kunlun::host_to_device));kunlun::check(kunlun::wait()); }
    double setup_ms=ms(t)-layout_check_ms; kunlun::Kernel legacy;kunlun::NativeQ4Workspace native;
    bool accelerated=false;
    auto run=[&] {
        if(mode=="native") accelerated=kunlun::native_q4_matmul(native,pd,dy.data(),dw.data(),dx.data(),k,n,batch,dtype=="f16"?kunlun::InputType::f16:kunlun::InputType::f32);
        else kunlun::matmul(legacy,pd,dy.data(),dw.data(),dx.data(),k,n,batch,mode=="cdnn"?kunlun::WeightType::rowwise_i8:kunlun::WeightType::q4_0,
            mode=="cdnn"?kunlun::packed_i8_scales(dw.data(),k,n):nullptr,mode=="cdnn"?kunlun::packed_i8_maxima(dw.data(),k,n):nullptr);
        if(mode!="native") { kunlun::select(pd);kunlun::check(kunlun::wait()); }
    };
    t=Clock::now(); run();double cold_ms=ms(t);
    for(int i=0;i<3;++i)run();
    setenv("K200_PROFILE_PHASE","steady",1);
    std::vector<double> times;
    for(int i=0;i<trials;++i){t=Clock::now();run();double elapsed=ms(t);times.push_back(elapsed);std::printf("TRIAL index=%d ms=%.9f\n",i,elapsed);}
    setenv("K200_PROFILE_PHASE","validation",1);
    std::vector<float> actual((size_t)n*batch);kunlun::check(kunlun::copy(actual.data(),dy.data(),actual.size()*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
    auto e=error(gold,actual); std::sort(times.begin(),times.end());double mean=std::accumulate(times.begin(),times.end(),0.)/times.size(),var=0;
    for(auto v:times)var+=(v-mean)*(v-mean);
    auto stats=native.statistics();
    std::printf("RESULT mode=%s pd=%d k=%d n=%d batch=%d dtype=%s pattern=%d accelerated=%d fallback_reason=%s trials=%d median_ms=%.9f min_ms=%.9f max_ms=%.9f stddev_ms=%.9f nmse=%.9g max_abs=%.9g max_rel=%.9g weight_bytes=%zu workspace_hbm=%zu workspace_l3=%zu hbm_allocated=%zu fixture_ms=%.6f setup_ms=%.6f cold_ms=%.6f cpu_ms=%.6f packing_ms=%.6f upload_ms=%.6f layout_check_ms=%.6f\n",
        mode.c_str(),pd,k,n,batch,dtype.c_str(),pattern,accelerated,accelerated?"none":kunlun::native_q4_reason(k,n,batch,dtype=="f16"?kunlun::InputType::f16:kunlun::InputType::f32),trials,times[times.size()/2],times.front(),times.back(),std::sqrt(var/times.size()),e.nmse,e.max_abs,e.max_rel,weights,stats.hbm_workspace_bytes,stats.l3_workspace_bytes,kunlun::allocated_bytes(pd)-before,fixture_ms,setup_ms,cold_ms,cpu_ms,packing_ms,upload_ms,layout_check_ms);
    if(auto emit=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_emit"))emit();
    if(auto memory=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_memory"))memory();
    require(e.nmse < (mode=="cdnn"?.01:accelerated?1e-5:1e-10));
}
