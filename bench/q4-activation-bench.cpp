#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <chrono>
#include <numeric>
#include <thread>
#include <condition_variable>
#include <mutex>
using namespace native_q4_bench;
using Clock=std::chrono::steady_clock;
static void checked(bool ok,int line) { if(!ok){std::fprintf(stderr,"ACTIVATION_CHECK line=%d\n",line);std::abort();} }
#define verify(x) checked((x),__LINE__)
static std::vector<unsigned char> read_device(const void * p,size_t n) {
    std::vector<unsigned char> v(n);kunlun::check(kunlun::copy(v.data(),p,n,kunlun::device_to_host));kunlun::check(kunlun::wait());return v;
}
static std::vector<double> bench_summary(std::vector<double> v) {
    verify(v.size()>1);std::sort(v.begin(),v.end());
    double mean=std::accumulate(v.begin(),v.end(),0.)/v.size(),var=0;
    for(double x:v)var+=(x-mean)*(x-mean);
    double median=(v[(v.size()-1)/2]+v[v.size()/2])*.5;
    return {mean,median,v.front(),v.back(),std::sqrt(var/(v.size()-1))};
}
int main(int argc,char ** argv) {
    verify(bench_summary({1,2,4,10})[1]==3);
    verify(argc==9);int pd=std::stoi(argv[1]),k=std::stoi(argv[2]),n=std::stoi(argv[3]),m=std::stoi(argv[4]);
    bool f16=std::string(argv[5])=="f16";int pattern=std::stoi(argv[6]),trials=std::stoi(argv[7]),consumers=std::stoi(argv[8]);
    verify(pd>=0&&pd<2&&m>0&&m<=1024&&trials>0&&trials<=31&&(consumers==2||consumers==3));
    int pattern_argument=pattern;verify((pattern>=0 && pattern<=2) || (pattern>=10 && pattern<=12));
    int skew=pattern>=10?(f16?2:4):0;pattern%=10;
    auto type=f16?kunlun::InputType::f16:kunlun::InputType::f32;int bytes=f16?2:4,n2=n==4096?11008:4096;
    Weights raw(k,n),raw2(k,n2);auto x=input(k,m,pattern);std::vector<_Float16> hx(x.size());
    if(f16)for(size_t i=0;i<x.size();++i){hx[i]=(_Float16)x[i];x[i]=(float)hx[i];}
    kunlun::Buffer dx(pd,x.size()*bytes+64),dy(pd,(size_t)m*std::max(n,n2)*4+512);
    kunlun::CommonQ4Weight w(pd,k,n),w2(pd,k,n2);w.upload(raw.q4.data(),raw.q4.size());w2.upload(raw2.q4.data(),raw2.q4.size());
    void * devx=(char *)dx.data()+skew;
    kunlun::select(pd);kunlun::check(kunlun::copy(devx,f16?(const void*)hx.data():x.data(),x.size()*bytes,kunlun::host_to_device));kunlun::check(kunlun::wait());
    auto * y=(float *)dy.data()+64;kunlun::CommonQ4Workspace legacy,shared;kunlun::CommonQ4ActivationLease lease;
    kunlun::common_q4_matmul(legacy,pd,y,w,devx,m,type);auto gold=read_device(y,(size_t)m*n*4);
    kunlun::common_q4_matmul(legacy,pd,y,w2,devx,m,type);auto gold2=read_device(y,(size_t)m*n2*4);
    verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,7,lease));
    verify(lease.matches(pd,k,m,type,7)&&lease.chunks()==(size_t)(m+127)/128);
    size_t quant_bytes=0,scale_bytes=0;
    for(size_t i=0;i<lease.chunks();++i) {
        auto c=lease.chunk(i);auto q=read_device(c.values,c.values_bytes),scale=read_device(c.scales,c.scales_bytes);
        std::vector<float> padded((size_t)c.padded_rows*k,0);
        for(int r=0;r<c.rows;++r)std::copy_n(x.data()+(size_t)(c.first+r)*k,k,padded.data()+(size_t)r*k);
        std::vector<_Float16> hp(padded.size());if(f16)for(size_t j=0;j<hp.size();++j)hp[j]=(_Float16)padded[j];
        kunlun::Buffer xp(pd,padded.size()*bytes),yp(pd,(size_t)c.padded_rows*n*4);
        kunlun::check(kunlun::copy(xp.data(),f16?(const void*)hp.data():padded.data(),xp.size(),kunlun::host_to_device));
        kunlun::common_q4_matmul(legacy,pd,yp.data(),w,xp.data(),c.padded_rows,type);
        auto v=legacy.quantization();verify(v.components==c.components&&v.kp==c.kp);
        verify(q==read_device(v.values,c.values_bytes)&&scale==read_device(v.scales,c.scales_bytes));
        quant_bytes+=q.size();scale_bytes+=scale.size();
    }
    verify(kunlun::common_q4_matmul_shared(shared,pd,y,w,lease,7));verify(read_device(y,gold.size())==gold);
    verify(kunlun::common_q4_matmul_shared(shared,pd,y,w2,lease,7));verify(read_device(y,gold2.size())==gold2);
    verify(!kunlun::common_q4_matmul_shared(shared,1-pd,y,w,lease,7));
    verify(!kunlun::common_q4_matmul_shared(shared,pd,y,w,lease,8));
    verify(!kunlun::common_q4_matmul_shared(legacy,pd,y,w,lease,7));
    verify(!kunlun::common_q4_matmul_shared(shared,pd,devx,w,lease,7));
    verify(!kunlun::common_q4_matmul_shared(shared,pd,const_cast<void*>(lease.chunk(0).values),w,lease,7));
    verify(!kunlun::common_q4_prepare(shared,pd,(char*)devx+1,k,m,type,99,lease));
    verify(!kunlun::common_q4_matmul_shared(shared,pd,(char*)y+2,w,lease,7));
    verify(!kunlun::common_q4_prepare(shared,pd,lease.chunk(0).values,k,m,type,99,lease));
    verify(!kunlun::common_q4_prepare(shared,pd,lease.chunk(0).scales,k,m,type,99,lease));
    verify(lease.matches(pd,k,m,type,7));
    lease.reset();uint64_t generation=100;
    auto old_call=[&](){for(int j=0;j<consumers;++j)kunlun::common_q4_matmul(legacy,pd,y,j==1?w2:w,devx,m,type);};
    auto new_call=[&](){++generation;verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,generation,lease));for(int j=0;j<consumers;++j)verify(kunlun::common_q4_matmul_shared(shared,pd,y,j==1?w2:w,lease,generation));lease.reset();};
    for(int i=0;i<3;++i){old_call();new_call();}
    auto before_old=legacy.activation_statistics(),before_new=shared.activation_statistics();std::vector<double> a,b;
    for(int i=0;i<trials;++i)for(int turn=0;turn<4;++turn){bool on=turn==1||turn==2;auto begin=Clock::now();if(on)new_call();else old_call();double ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();(on?b:a).push_back(ms);}
    auto after_old=legacy.activation_statistics(),after_new=shared.activation_statistics();
    verify(after_old.quantized_chunks-before_old.quantized_chunks==(size_t)trials*2*consumers*((m+127)/128));
    verify(after_new.quantized_chunks-before_new.quantized_chunks==(size_t)trials*2*((m+127)/128));
    // Same-address writes require a new owner generation and invalidate old leases.
    kunlun::CommonQ4ActivationLease stale,fresh;
    verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,900,stale));
    for(size_t i=0;i<x.size();++i)x[i]=(i%7-3.0f)*.125f;
    if(f16)for(size_t i=0;i<x.size();++i)hx[i]=(_Float16)x[i];
    kunlun::check(kunlun::copy(devx,f16?(const void*)hx.data():x.data(),x.size()*bytes,kunlun::host_to_device));
    verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,901,fresh));
    verify(!stale.matches(pd,k,m,type,900)&&!kunlun::common_q4_matmul_shared(shared,pd,y,w,stale,900));
    verify(kunlun::common_q4_matmul_shared(shared,pd,y,w,fresh,901));auto changed=read_device(y,gold.size());
    kunlun::common_q4_matmul(legacy,pd,y,w,devx,m,type);verify(read_device(y,gold.size())==changed);
    kunlun::common_q4_matmul(shared,pd,y,w,devx,m,type);verify(!fresh.matches(pd,k,m,type,901));
    int too_many=(int)((16*1024*1024/((size_t)128*w.info().kp+1024)+1)*128);
    verify(!kunlun::common_q4_prepare(shared,pd,devx,k,too_many,type,902,fresh));
    verify(!kunlun::common_q4_prepare(shared,pd,devx,k,m,type,0,fresh));
    verify(!kunlun::common_q4_prepare(shared,pd,devx,k,m,(kunlun::InputType)99,902,fresh));
    {kunlun::CommonQ4Workspace short_lived;verify(kunlun::common_q4_prepare(short_lived,pd,devx,k,m,type,903,fresh));}
    verify(!fresh.matches(pd,k,m,type,903));fresh.reset();
    std::mutex gate;std::condition_variable cv;int stage=0;
    std::thread first([&](){
        kunlun::CommonQ4ActivationLease a;verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,1000,a));
        {std::unique_lock<std::mutex> lock(gate);stage=1;cv.notify_all();cv.wait(lock,[&](){return stage>=2;});}
        verify(!kunlun::common_q4_matmul_shared(shared,pd,y,w,a,1000));
        {std::lock_guard<std::mutex> lock(gate);stage=3;cv.notify_all();}
    });
    std::thread second([&](){
        {std::unique_lock<std::mutex> lock(gate);cv.wait(lock,[&](){return stage>=1;});}
        kunlun::CommonQ4ActivationLease a;verify(kunlun::common_q4_prepare(shared,pd,devx,k,m,type,1001,a));
        {std::unique_lock<std::mutex> lock(gate);stage=2;cv.notify_all();cv.wait(lock,[&](){return stage>=3;});}
        verify(kunlun::common_q4_matmul_shared(shared,pd,y,w,a,1001));
    });
    first.join();second.join();verify(read_device(y,gold.size())==changed);
    if(k==512 && n==512 && m==1) {
        // A borrowed weight must not alias the recorded raw activation source.
        kunlun::Buffer alias(pd,w.info().payload_bytes);
        kunlun::CommonQ4Weight overlap(pd,k,n,alias.data(),alias.size());overlap.upload(raw.q4.data(),raw.q4.size());
        kunlun::check(kunlun::copy(alias.data(),f16?(const void*)hx.data():x.data(),x.size()*bytes,kunlun::host_to_device));kunlun::check(kunlun::wait());
        verify(kunlun::common_q4_prepare(shared,pd,alias.data(),k,m,type,2000,fresh));
        verify(!kunlun::common_q4_matmul_shared(shared,pd,y,overlap,fresh,2000));verify(read_device(y,gold.size())==changed);fresh.reset();
    }
    auto av=bench_summary(a),bv=bench_summary(b);auto oldmem=legacy.statistics(),newmem=shared.statistics();
    std::printf("ACTIVATION_RESULT {\"pd\":%d,\"k\":%d,\"n\":%d,\"m\":%d,\"f16\":%s,\"pattern\":%d,\"pattern_argument\":%d,\"input_skew_bytes\":%d,\"consumers\":%d,\"samples\":%zu,\"old_ms\":[",pd,k,n,m,f16?"true":"false",pattern,pattern_argument,skew,consumers,a.size());
    for(size_t i=0;i<av.size();++i)std::printf("%s%.7f",i?",":"",av[i]);std::printf("],\"shared_ms\":[");for(size_t i=0;i<bv.size();++i)std::printf("%s%.7f",i?",":"",bv[i]);
    std::printf("],\"old_samples_ms\":[");for(size_t i=0;i<a.size();++i)std::printf("%s%.9f",i?",":"",a[i]);
    std::printf("],\"shared_samples_ms\":[");for(size_t i=0;i<b.size();++i)std::printf("%s%.9f",i?",":"",b[i]);
    std::printf("],\"old_quant_chunks\":%zu,\"shared_quant_chunks\":%zu,\"old_quant_kernels\":%zu,\"shared_quant_kernels\":%zu,\"old_hbm_workspace\":%zu,\"shared_hbm_workspace\":%zu,\"quant_bytes_checked\":%zu,\"scale_bytes_checked\":%zu,\"bitwise\":true,\"lifetime_checks\":true}\n",after_old.quantized_chunks-before_old.quantized_chunks,after_new.quantized_chunks-before_new.quantized_chunks,after_old.quant_kernel_launches-before_old.quant_kernel_launches,after_new.quant_kernel_launches-before_new.quant_kernel_launches,oldmem.hbm_workspace_bytes,newmem.hbm_workspace_bytes,quant_bytes,scale_bytes);
}
