#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0,k=argc>2?std::atoi(argv[2]):4096,n=argc>3?std::atoi(argv[3]):4096,batch=argc>4?std::atoi(argv[4]):1;
    kunlun::require(pd>=0 && pd<2 && k>0 && k<=16384 && k%32==0 && n>=16 && n<=32768 && batch>0 && batch<=32);
    std::vector<float>w((size_t)k*n),x((size_t)k*batch),y((size_t)n*batch),legacy(y.size());
    std::vector<unsigned char>q((size_t)k*n/32*18);
    for(int row=0;row<n;++row) for(int block=0;block<k/32;++block) {
        float d=(block%3==0?.015625f:block%3==1?.03125f:.046875f)*(block%2?-1:1);
        _Float16 h=(_Float16)d;unsigned char *p=q.data()+((size_t)row*(k/32)+block)*18;std::memcpy(p,&h,2);
        for(int j=0;j<32;++j)w[(size_t)row*k+block*32+j]=((row+j)%16-8)*d;
        for(int j=0;j<16;++j)p[2+j]=((row+j)%16)|(((row+j+16)%16)<<4);
    }
    for(size_t j=0;j<x.size();++j)x[j]=std::sin(j*.017f);
    std::vector<double>gold((size_t)batch*16);
    for(int b=0;b<batch;++b)for(int r=0;r<16;++r)for(int j=0;j<k;++j)gold[b*16+r]+=double(w[r*k+j])*x[b*k+j];
    kunlun::Buffer qw(pd,q.size()),iw(pd,kunlun::packed_i8_size(k,n)),dx(pd,x.size()*4),dy(pd,y.size()*4);
    kunlun::Kernel direct,fast;
    kunlun::select(pd);kunlun::check(kunlun::copy(qw.data(),q.data(),q.size(),kunlun::host_to_device));
    kunlun::check(kunlun::copy(dx.data(),x.data(),x.size()*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
    kunlun::upload_i8_rows(pd,iw.data(),k,n,0,n,w.data());
    for(int mode=0;mode<3;++mode) {
        setenv("KUNLUN_GEMV",mode==2?"fused":"legacy",1);
        auto run=[&]{
            kunlun::matmul(mode?fast:direct,pd,dy.data(),mode?iw.data():qw.data(),dx.data(),k,n,batch,
                mode?kunlun::WeightType::rowwise_i8:kunlun::WeightType::q4_0,
                mode?kunlun::packed_i8_scales(iw.data(),k,n):nullptr,mode?kunlun::packed_i8_maxima(iw.data(),k,n):nullptr);
            kunlun::select(pd);kunlun::check(kunlun::wait());
        };
        run();std::vector<double>times;int loops=mode?20:1;
        for(int group=0;group<5;++group){auto start=std::chrono::steady_clock::now();for(int i=0;i<loops;++i)run();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/loops);}
        kunlun::check(kunlun::copy(y.data(),dy.data(),y.size()*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
        double max_abs=0,max_rel=0,squared=0,energy=0,legacy_error=0,legacy_energy=0,max_delta=0;
        for(int b=0;b<batch;++b)for(int r=0;r<n;++r){size_t j=(size_t)b*n+r;double ref=gold[b*16+r%16],error=y[j]-ref;
            if(!std::isfinite(y[j]))return 1;
            max_abs=std::max(max_abs,std::abs(error));max_rel=std::max(max_rel,std::abs(error)/std::max(std::abs(ref),1e-20));squared+=error*error;energy+=ref*ref;
            if(!mode && std::abs(error)>.001*(1+std::abs(ref)))return 1;
            if(mode==2){double e=y[j]-legacy[j];legacy_error+=e*e;legacy_energy+=double(legacy[j])*legacy[j];max_delta=std::max(max_delta,std::abs(e));}
        }
        if(mode==1)legacy=y;
        if(mode==2 && legacy_error/std::max(legacy_energy,1e-30)>1e-8)return 1;
        std::sort(times.begin(),times.end());
        std::printf("gemv mode=%s pd=%d k=%d n=%d batch=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f cpu_max_abs=%g cpu_max_rel=%g cpu_nmse=%g legacy_max_abs=%g legacy_nmse=%g q4_bytes=%zu i8_bytes=%zu PASS\n",mode==0?"q4":mode==1?"two_fc":"fused",pd,k,n,batch,times[2],times.front(),times.back(),max_abs,max_rel,squared/std::max(energy,1e-30),max_delta,legacy_error/std::max(legacy_energy,1e-30),q.size(),kunlun::packed_i8_size(k,n));
    }
}
