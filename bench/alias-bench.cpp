#include <kunlun/k200.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0,width=argc>2?std::atoi(argv[2]):4096,batch=argc>3?std::atoi(argv[3]):6;
    kunlun::require(pd>=0 && pd<2 && width>0 && width<=16384 && batch>0 && batch<=32);
    int count=width*batch;
    std::vector<float>a(count),b(count),actual(count);
    for(int j=0;j<count;++j)a[j]=std::sin(j*.021f);
    for(int j=0;j<count;++j)b[j]=std::cos(j*.04f);
    kunlun::Buffer da(pd,count*4),db(pd,count*4);
    kunlun::Kernel kernel;
    kunlun::select(pd);
    double maximum=0;
    for(auto op:{kunlun::ElementwiseOp::add,kunlun::ElementwiseOp::multiply}) {
      for(int bn:{width,count}) for(int alias:{0,1}) {
        for(int repeat=0;repeat<100;++repeat) {
            kunlun::check(kunlun::copy(da.data(),a.data(),count*4,kunlun::host_to_device));
            kunlun::check(kunlun::copy(db.data(),b.data(),bn*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
            void *output=alias?db.data():da.data();
            kunlun::elementwise(kernel,pd,output,da.data(),db.data(),count,width,bn,op);
            kunlun::check(kunlun::copy(actual.data(),output,count*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            for(int j=0;j<count;++j) {
                float ref=op==kunlun::ElementwiseOp::add?a[j]+b[j%bn]:a[j]*b[j%bn];
                double error=std::abs(actual[j]-ref);maximum=std::max(maximum,error);
                if(!std::isfinite(actual[j]) || error>1e-6*(1+std::abs(ref))) {
                    std::fprintf(stderr,"FAIL pd=%d op=%d repeat=%d index=%d got=%g ref=%g\n",pd,(int)op,repeat,j,actual[j],ref);return 1;
                }
            }
        }
      }
    }
    std::printf("alias broadcast pd=%d width=%d batch=%d repeats=100 max_abs=%g PASS\n",pd,width,batch,maximum);
}
