#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    int pd=argc > 1 ? std::atoi(argv[1]) : 0;
    int width=argc > 2 ? std::atoi(argv[2]) : 16;
    int rows=argc > 3 ? std::atoi(argv[3]) : 1;
    int stride=argc > 4 ? std::atoi(argv[4]) : 32;
    int column=argc > 5 ? std::atoi(argv[5]) : 8;
    kunlun::require(pd >= 0 && pd < 2 && width > 0 && rows > 0 && stride >= width && column >= 0 && column <= stride-width);
    kunlun::require(stride <= 100000 && rows <= 32);
    std::vector<float> src(width*rows), initial(stride*rows+128,-123.25f), actual(initial.size()), gold=initial;
    for (size_t i=0;i<src.size();++i) src[i]=(int(i%65521)-30000)*.125f;
    for (int r=0;r<rows;++r) std::copy_n(src.data()+r*width,width,gold.data()+64+r*stride+column);
    kunlun::Buffer input(pd,src.size()*4),output(pd,initial.size()*4);
    kunlun::Kernel kernel;
    kunlun::select(pd);
    kunlun::check(kunlun::copy(input.data(),src.data(),src.size()*4,kunlun::host_to_device));
    kunlun::check(kunlun::copy(output.data(),initial.data(),initial.size()*4,kunlun::host_to_device));
    kunlun::check(kunlun::wait());
    auto run=[&] {
        kunlun::copy_rows(kernel,pd,(float *)output.data()+64,input.data(),width,rows,stride,column);
        kunlun::check(kunlun::wait());
    };
    run(); run();
    std::vector<double> groups;
    for (int g=0;g<5;++g) {
        auto start=std::chrono::steady_clock::now();
        for (int i=0;i<20;++i) run();
        groups.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20);
    }
    kunlun::check(kunlun::copy(actual.data(),output.data(),actual.size()*4,kunlun::device_to_host));
    kunlun::check(kunlun::wait());
    bool ok=std::memcmp(actual.data(),gold.data(),gold.size()*4)==0;
    std::sort(groups.begin(),groups.end());
    std::printf("copy_rows pd=%d width=%d rows=%d stride=%d column=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f bitwise_equal=%d %s\n",pd,width,rows,stride,column,groups[2],groups.front(),groups.back(),ok,ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
