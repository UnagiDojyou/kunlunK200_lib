#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <chrono>
#include <string>
#include <numeric>
#include <dlfcn.h>
using namespace native_q4_bench;
static void checked(bool ok,int line){if(!ok){std::fprintf(stderr,"COMMON BENCH assertion line=%d\n",line);std::abort();}}
#define require(x) checked((x),__LINE__)
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char ** argv){
    require(argc==9);int pd=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]),m=std::stoi(argv[4]);
    bool f16=std::string(argv[5])=="f16";int pattern=std::stoi(argv[6]),trials=std::stoi(argv[7]);std::string mode=argv[8];
    require(mode=="common");
    require(pd>=0&&pd<2&&kunlun::common_q4_supported(k,n)&&m>0&&m<=1024&&trials>0&&trials<=31);
    const int blocks=(k+31)/32;
    std::vector<unsigned char> raw((size_t)n*blocks*18);uint32_t rng=0x7340d123;
    if(const char * file=std::getenv("KUNLUN_Q4_WEIGHT_FILE")){std::ifstream in(file,std::ios::binary|std::ios::ate);require(in.good()&&(size_t)in.tellg()==raw.size());in.seekg(0);in.read((char*)raw.data(),raw.size());require(in.good());}
    else for(int r=0;r<n;++r)for(int b=0;b<blocks;++b){auto p=raw.data()+((size_t)r*blocks+b)*18;uint32_t h=random(rng);float d=std::ldexp((8+h%9)/1024.f,int((h>>4)%7)-3);if(h&0x80000000)d=-d;if((r*13+b)%257==0)d=0;_Float16 hd=(_Float16)d;std::memcpy(p,&hd,2);for(int j=0;j<16;++j)p[2+j]=random(rng);}
    auto x=input(k,m,pattern);
    if(const char * file=std::getenv("KUNLUN_Q4_ACTIVATION_FILE")){std::ifstream in(file,std::ios::binary);in.read((char*)x.data(),x.size()*4);require(in.good());}
    std::vector<_Float16> hx(x.size());
    if(f16)for(size_t i=0;i<x.size();++i){hx[i]=(_Float16)x[i];x[i]=(float)hx[i];}
    setenv("K200_PROFILE_PHASE","load",1);
    kunlun::Buffer dx(pd,x.size()*(f16?2:4)),dy(pd,((size_t)m*n+128)*4);
    kunlun::select(pd);kunlun::check(kunlun::copy(dx.data(),f16?(const void*)hx.data():x.data(),dx.size(),kunlun::host_to_device));kunlun::check(kunlun::wait());
    std::vector<float> poison(dy.size()/4,-1234567.f);kunlun::check(kunlun::copy(dy.data(),poison.data(),dy.size(),kunlun::host_to_device));
    float * out=(float*)dy.data()+64;
    auto setup=Clock::now();kunlun::CommonQ4Weight weight(pd,k,n);weight.upload(raw.data(),raw.size());double setup_ms=ms(setup);
    kunlun::CommonQ4Workspace workspace;
    auto invoke=[&](){kunlun::common_q4_matmul(workspace,pd,out,weight,dx.data(),m,f16?kunlun::InputType::f16:kunlun::InputType::f32);};
    setenv("K200_PROFILE_PHASE","warmup",1);
    auto t=Clock::now();invoke();double cold=ms(t);invoke();
    std::vector<float> values(poison.size());kunlun::check(kunlun::copy(values.data(),dy.data(),dy.size(),kunlun::device_to_host));kunlun::check(kunlun::wait());
    auto stable=values;std::vector<double> samples;
    for(int i=0;i<trials;++i){setenv("K200_PROFILE_PHASE","steady",1);t=Clock::now();invoke();samples.push_back(ms(t));setenv("K200_PROFILE_PHASE","validation",1);kunlun::check(kunlun::copy(values.data(),dy.data(),dy.size(),kunlun::device_to_host));require(values==stable);}
    for(int i=0;i<64;++i)require(values[i]==-1234567.f&&values[values.size()-1-i]==-1234567.f);
    auto info=weight.info();auto view=workspace.quantization();int kp=info.kp;
    std::vector<int8_t> q((size_t)(m==1?4:128)*kp+1024);std::vector<float> scales(128*16);
    if(view.components)kunlun::check(kunlun::copy(q.data(),view.values,m==1?4*kp+1024:128*kp,kunlun::device_to_host));
    if(view.components)kunlun::check(kunlun::copy(scales.data(),view.scales,m==1?64:128*64,kunlun::device_to_host));kunlun::check(kunlun::wait());
    std::vector<float> qa((size_t)m*k);size_t quant_mismatch=0;int first=m-view.rows;
    for(int row=0;row<m;++row){
        if(m==1 && !view.components){qa=x;continue;}
        if(m==1){std::vector<float> padded(kp,0);std::copy(x.begin(),x.end(),padded.begin());Activation cpu(padded,kp,1,8,2,false,true);
            for(int j=0;j<k;++j){int i=(j/32)*64+j%32;qa[j]=q[i]*scales[0]+q[2*kp+i]*scales[1];quant_mismatch+=q[i]!=cpu.linear[j];quant_mismatch+=q[2*kp+i]!=cpu.linear[kp+j];}
        }else{float maximum=0;for(int j=0;j<k;++j)maximum=std::max(maximum,std::abs(x[(size_t)row*k+j]));if(maximum==0)maximum=1;
            float inv=127.f/maximum,s=maximum/127.f;
            for(int j=0;j<k;++j){int code=rounded(x[(size_t)row*k+j]*inv,127);float scale=s;
                if(row>=first){int actual=q[(size_t)(row-first)*kp+j];quant_mismatch+=actual!=code;code=actual;scale=scales[(row-first)*16];}
                qa[(size_t)row*k+j]=code*scale;
            }
        }
    }
    // Independent raw Q4 decoding and double reference. No common-pack mapping here.
    double impl_sq=0,quant_sq=0,total_sq=0,energy=0,repr_energy=0,maxabs=0;
    #pragma omp parallel for num_threads(6) reduction(+:impl_sq,quant_sq,total_sq,energy,repr_energy) reduction(max:maxabs)
    for(int col=0;col<n;++col)for(int row=0;row<m;++row){double original=0,represented=0;
        for(int b=0;b<blocks;++b){auto p=raw.data()+((size_t)col*blocks+b)*18;double scale=half(p);double a=0,v=0;
            #pragma omp simd reduction(+:a,v)
            for(int j=0;j<std::min(32,k-b*32);++j){int wi=nibble(p,j);a+=double(x[(size_t)row*k+b*32+j])*wi;v+=double(qa[(size_t)row*k+b*32+j])*wi;}
            original+=scale*a;represented+=scale*v;
        }
        double actual=values[64+(size_t)row*n+col],e=actual-represented;require(std::isfinite(actual));
        impl_sq+=e*e;quant_sq+=(represented-original)*(represented-original);total_sq+=(actual-original)*(actual-original);energy+=original*original;repr_energy+=represented*represented;maxabs=std::max(maxabs,std::abs(e));
    }
    double impl=impl_sq/std::max(repr_energy,1e-30);std::fprintf(stderr,"VALIDATION impl=%.12g quant_mismatch=%zu first=%d actual0=%.12g\n",impl,quant_mismatch,first,values[64]);require(impl<=1e-10);
    auto ws=workspace.statistics();auto st=weight.statistics();require(st.uploads==1&&ws.calls==(size_t)trials+2);
    double mean=std::accumulate(samples.begin(),samples.end(),0.)/samples.size(),variance=0;for(auto v:samples)variance+=(v-mean)*(v-mean);auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    std::printf("COMMON_RESULT {\"pd\":%d,\"k\":%d,\"n\":%d,\"m\":%d,\"dtype\":\"%s\",\"pattern\":%d,\"implementation_nmse\":%.12g,\"quantization_nmse\":%.12g,\"total_nmse\":%.12g,\"max_abs\":%.12g,\"cpu_quant_differences\":%zu,\"setup_ms\":%.6f,\"pack_ms\":%.6f,\"upload_ms\":%.6f,\"cold_ms\":%.6f,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"std_ms\":%.6f,\"weight_bytes\":%zu,\"uploads\":%zu,\"hbm_workspace\":%zu,\"l3_workspace\":%zu,\"slice_bytes\":%zu,\"paths\":[",pd,k,n,m,f16?"f16":"f32",pattern,impl,quant_sq/std::max(energy,1e-30),total_sq/std::max(energy,1e-30),maxabs,quant_mismatch,setup_ms,st.pack_ms,st.upload_ms,cold,sorted[sorted.size()/2],sorted.front(),sorted.back(),std::sqrt(variance/samples.size()),info.payload_bytes,st.uploads,ws.hbm_workspace_bytes,ws.l3_workspace_bytes,ws.slice_bytes);
    for(int i=0;i<8;++i)std::printf("%s%zu",i?",":"",ws.paths[i]);std::printf("],\"samples_ms\":[");for(size_t i=0;i<samples.size();++i)std::printf("%s%.6f",i?",":"",samples[i]);std::puts("],\"correct\":true}");
    if(auto emit=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_emit"))emit();
    if(auto memory=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_memory"))memory();
}
