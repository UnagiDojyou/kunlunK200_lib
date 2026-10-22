#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){
    kunlun::require(argc==4);int pd=std::atoi(argv[1]),width=std::atoi(argv[2]),rows=std::atoi(argv[3]);
    kunlun::require(pd>=0&&pd<2&&width>0&&width<=16384&&rows>0&&rows<=256);
    int n=width*rows;kunlun::Buffer a(pd,(size_t)n*4),b(pd,(size_t)n*4),y(pd,(size_t)n*4);
    std::vector<float> av(n),bv(n),reference(n),actual(n),gold(n);
    for(int i=0;i<n;++i){av[i]=std::sin(i*.13f);bv[i]=std::cos((i%width)*.17f);}
    kunlun::Kernel kernel;
    for(int op:{0,1})for(int alias:{0,1,2}){
        for(int i=0;i<n;++i)reference[i]=op?av[i]*bv[i%width]:av[i]+bv[i%width];
        for(int vector:{0,1}){
            setenv("KUNLUN_BROADCAST_VECTOR",vector?"1":"0",1);
            auto run=[&](){
                kunlun::select(pd);kunlun::check(kunlun::copy(a.data(),av.data(),(size_t)n*4,kunlun::host_to_device));kunlun::check(kunlun::copy(b.data(),bv.data(),(size_t)n*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
                void* out=alias==1?a.data():alias==2?b.data():y.data();
                auto t=Clock::now();kunlun::elementwise(kernel,pd,out,a.data(),b.data(),n,width,width,(kunlun::ElementwiseOp)op);
                double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
                kunlun::check(kunlun::copy(actual.data(),out,(size_t)n*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
                kunlun::require(actual==reference);return elapsed;
            };
            run();std::vector<double> times;for(int trial=0;trial<5;++trial)times.push_back(run());
            if(!vector)gold=actual;else kunlun::require(std::memcmp(gold.data(),actual.data(),actual.size()*4)==0);
            std::sort(times.begin(),times.end());
            std::printf("BROADCAST {\"pd\":%d,\"width\":%d,\"rows\":%d,\"op\":%d,\"alias\":%d,\"vector\":%d,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"bitwise_correct\":true}\n",pd,width,rows,op,alias,vector,times[2],times[0],times[4]);
        }
    }
}
