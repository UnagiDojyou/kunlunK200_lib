#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <memory>
using namespace native_q4_bench;
int main() {
    require(!kunlun::native_q4_layout_supported(31,256));
    require(!kunlun::native_q4_layout_supported(17408,256));
    require(!kunlun::native_q4_layout_supported(4096,0));
    for(int pd=0;pd<2;++pd) {
        kunlun::NativeQ4Workspace first,second;
        for(int index=0;index<6;++index) {
            int widths[]={4096,96,544,32,4096,96};
            int k=widths[index],n=index%2?384:512;
            Weights w(k,n);auto x=input(k,1,3);auto gold=reference(w.decoded,x,k,n,1);
            kunlun::Buffer dw(pd,kunlun::native_q4_size(k,n)),dx(pd,k*4+64),dy(pd,n*4+64);
            kunlun::select(pd);kunlun::upload_native_q4(pd,dw.data(),w.q4.data(),k,n);
            kunlun::check(kunlun::copy(dx.data(),x.data(),k*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
            auto & workspace=index%3?first:second;
            require(kunlun::native_q4_matmul(workspace,pd,dy.data(),dw.data(),dx.data(),k,n));
            std::vector<float> y(n);kunlun::check(kunlun::copy(y.data(),dy.data(),n*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            auto e=error(gold,y);require(e.nmse<1e-6);
            // Non-SD alignment must run exact CL; it must not silently skip output.
            require(!kunlun::native_q4_matmul(workspace,pd,(char *)dy.data()+4,dw.data(),dx.data(),k,n));
            kunlun::check(kunlun::copy(y.data(),(char *)dy.data()+4,n*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            auto f=error(gold,y);require(f.nmse<1e-10);
            kunlun::check(kunlun::copy((char *)dx.data()+4,x.data(),k*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
            require(!kunlun::native_q4_matmul(workspace,pd,dy.data(),dw.data(),(char *)dx.data()+4,k,n));
            kunlun::check(kunlun::copy(y.data(),dy.data(),n*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            require(error(gold,y).nmse<1e-10);
            std::vector<unsigned short> half(k);std::vector<float> half_x(k);
            for(int j=0;j<k;++j){_Float16 h=(_Float16)x[j];std::memcpy(&half[j],&h,2);half_x[j]=(float)h;}
            auto half_gold=reference(w.decoded,half_x,k,n,1);
            kunlun::check(kunlun::copy((char *)dx.data()+2,half.data(),k*2,kunlun::host_to_device));kunlun::check(kunlun::wait());
            require(!kunlun::native_q4_matmul(workspace,pd,dy.data(),dw.data(),(char *)dx.data()+2,k,n,1,kunlun::InputType::f16));
            kunlun::check(kunlun::copy(y.data(),dy.data(),n*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
            require(error(half_gold,y).nmse<1e-10);
            std::printf("STATE PASS pd=%d index=%d k=%d n=%d sd_nmse=%.9g fallback_nmse=%.9g\n",pd,index,k,n,e.nmse,f.nmse);
        }
    }
}
