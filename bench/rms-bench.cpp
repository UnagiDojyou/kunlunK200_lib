#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0,width=argc>2?std::atoi(argv[2]):4096,rows=argc>3?std::atoi(argv[3]):1;
    kunlun::require(pd>=0 && pd<2 && width>=256 && width<=16384 && width%32==0 && rows>0 && rows<=32);
    int count=width*rows;float epsilon=1e-5f;
    const char *mode=std::getenv("KUNLUN_RMS_NORM");bool legacy=mode && std::strcmp(mode,"legacy")==0;
    double tolerance=legacy?1e-4:1e-5;
    std::vector<float>input(count),actual(count),gold(count);kunlun::Buffer x(pd,count*4),y(pd,count*4);kunlun::Kernel kernel;
    double max_abs=0,max_rel=0,error=0,energy=0;
    for(int pattern=0;pattern<4;++pattern) {
        for(int j=0;j<count;++j) input[j]=pattern==0?0:pattern==1?std::sin(j*.017f):pattern==2?(j%5?-1e-3f:1e3f):std::sin(j*.017f)*1e-20f;
        for(int row=0;row<rows;++row) {
            double sum=0;for(int j=0;j<width;++j) {double value=input[row*width+j];sum+=value*value;}
            float scale=1.0f/std::sqrt(float(sum/width)+epsilon);
            for(int j=0;j<width;++j)gold[row*width+j]=input[row*width+j]*scale;
        }
        for(int alias=0;alias<2;++alias) {
            kunlun::select(pd);kunlun::check(kunlun::copy(x.data(),input.data(),count*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
            void *out=alias?x.data():y.data();
            kunlun::elementwise(kernel,pd,out,x.data(),x.data(),count,width,count,kunlun::ElementwiseOp::rms_norm,epsilon);
            kunlun::check(kunlun::copy(actual.data(),out,count*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            for(int j=0;j<count;++j) {
                double e=std::abs(actual[j]-gold[j]);max_abs=std::max(max_abs,e);max_rel=std::max(max_rel,e/std::max(double(std::abs(gold[j])),1e-30));error+=e*e;energy+=double(gold[j])*gold[j];
                if(!std::isfinite(actual[j]) || e>tolerance+tolerance*std::abs(gold[j])) {std::fprintf(stderr,"FAIL pattern=%d alias=%d index=%d got=%g ref=%g\n",pattern,alias,j,actual[j],gold[j]);return 1;}
            }
        }
    }
    if(error/std::max(energy,1e-30)>(legacy?1e-8:1e-10))return 1;
    for(int j=0;j<count;++j)input[j]=std::sin(j*.017f);
    kunlun::check(kunlun::copy(x.data(),input.data(),count*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
    auto run=[&]{kunlun::elementwise(kernel,pd,y.data(),x.data(),x.data(),count,width,count,kunlun::ElementwiseOp::rms_norm,epsilon);};
    run();run();std::vector<double>times;
    for(int group=0;group<5;++group){auto start=std::chrono::steady_clock::now();for(int i=0;i<20;++i)run();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20);}
    std::sort(times.begin(),times.end());
    std::printf("rms pd=%d width=%d rows=%d max_abs=%g max_rel=%g nmse=%g median_ms=%.6f min_ms=%.6f max_ms=%.6f PASS\n",pd,width,rows,max_abs,max_rel,error/std::max(energy,1e-30),times[2],times.front(),times.back());
}
