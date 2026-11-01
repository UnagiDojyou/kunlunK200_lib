#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace kunlun;
static double elapsed(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count(); }
int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0;
    Kernel kernel;
    for(int k:{16,80,2048,4096}) {
        const int n=k<128?3:128;
        const uint32_t guard=0xa5b6c7d8;
        std::vector<uint32_t> input((size_t)k*n+128,guard),output((size_t)k*512+128,guard),ref((size_t)k*512);
        const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x7fc01234,0x7fa01234,1,0x7f7fffff};
        for(size_t j=0;j<(size_t)k*n;++j)input[64+j]=(j%17==0)?special[j%8]:(uint32_t)(j*1664525u+1013904223u);
        std::vector<int32_t> ids(512+128,0x12345678);
        Buffer x(pd,input.size()*4),y(pd,output.size()*4),index(pd,ids.size()*4);
        auto *dx=(uint32_t*)x.data()+64,*dy=(uint32_t*)y.data()+64;auto *di=(int32_t*)index.data()+64;
        check(copy(x.data(),input.data(),input.size()*4,host_to_device));
        for(int m:{1,3,128,512}) {
            for(int i=0;i<m;++i)ids[64+i]=i%3==0?n-1:i%3==1?0:(i*17)%n;
            auto reset=[&]{std::fill(output.begin(),output.end(),guard);check(copy(y.data(),output.data(),output.size()*4,host_to_device));};reset();
            check(copy(index.data(),ids.data(),ids.size()*4,host_to_device));
            for(int i=0;i<m;++i)std::memcpy(ref.data()+(size_t)i*k,input.data()+64+(size_t)ids[64+i]*k,k*4);
            auto run=[&]{return get_rows_f32(kernel,pd,dy,(size_t)m*k*4,dx,(size_t)n*k*4,di,m*4,k,n,m);};require(run());
            check(copy(output.data(),y.data(),output.size()*4,device_to_host));require(!std::memcmp(output.data()+64,ref.data(),(size_t)m*k*4));
            for(size_t i=0;i<64;++i)require(output[i]==guard);
            for(size_t i=(size_t)m*k+64;i<output.size();++i)require(output[i]==guard);
            std::vector<double>times;
            for(int r=0;r<3;++r){auto t=std::chrono::steady_clock::now();require(run());times.push_back(elapsed(t));}
            std::sort(times.begin(),times.end());std::printf("F32_ROWS pd=%d K=%d N=%d M=%d bitwise=1 gpu_ms=%.6f\n",pd,k,n,m,times[1]);
            for(int bad:{-1,n}) {
                reset();auto saved=ids[64];ids[64]=bad;check(copy(di,ids.data()+64,m*4,host_to_device));require(!run());ids[64]=saved;
                check(copy(output.data(),y.data(),output.size()*4,device_to_host));for(auto v:output)require(v==guard);
            }
            check(copy(di,ids.data()+64,m*4,host_to_device));
            require(!get_rows_f32(kernel,pd,dy,(size_t)m*k*4-1,dx,(size_t)n*k*4,di,m*4,k,n,m));
            require(!get_rows_f32(kernel,pd,dy,(size_t)m*k*4,dx,(size_t)n*k*4-1,di,m*4,k,n,m));
            require(!get_rows_f32(kernel,pd,dy,(size_t)m*k*4,dx,(size_t)n*k*4,di,m*4-1,k,n,m));
            require(!get_rows_f32(kernel,pd,dx,(size_t)n*k*4,dx,(size_t)n*k*4,di,m*4,k,n,m));
        }
        require(!get_rows_f32(kernel,pd,dy,(size_t)k*512*4,dx,(size_t)n*k*4,di,512*4,k,n,513));
        require(!get_rows_f32(kernel,pd,dy,(size_t)k*512*4,dx,(size_t)n*k*4,di,512*4,k-1,n,1));
        std::vector<uint32_t>back(input.size());check(copy(back.data(),x.data(),back.size()*4,device_to_host));require(input==back);
    }
    std::puts("F32_GET_ROWS_PASS");
}
