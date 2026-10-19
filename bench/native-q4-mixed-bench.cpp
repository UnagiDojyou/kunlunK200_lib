#include <kunlun/k200.h>
#include "native-q4-reference.h"
using namespace native_q4_bench;
int main() {
    for (int pd=0;pd<2;++pd) for (int legacy_first=0;legacy_first<2;++legacy_first) {
        kunlun::NativeQ4Workspace native;kunlun::Kernel legacy;
        for (int rows : {5504,11008}) {
            Weights w(4096,rows);auto x=input(w.k,1,3);auto gold=reference(w.decoded,x,w.k,w.n,1);
            kunlun::Buffer dw(pd,kunlun::native_q4_size(w.k,w.n)),di8(pd,kunlun::packed_i8_size(w.k,w.n));
            kunlun::Buffer dx(pd,w.k*4),dy(pd,w.n*4);
            kunlun::select(pd);kunlun::upload_native_q4(pd,dw.data(),w.q4.data(),w.k,w.n);
            kunlun::upload_i8_rows(pd,di8.data(),w.k,w.n,0,w.n,w.decoded.data());
            kunlun::check(kunlun::copy(dx.data(),x.data(),x.size()*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
            std::vector<float> y(w.n);
            for (int step=0;step<6;++step) {
                bool cdnn=(step+legacy_first)%2;
                if (cdnn) {
                    kunlun::matmul(legacy,pd,dy.data(),di8.data(),dx.data(),w.k,w.n,1,kunlun::WeightType::rowwise_i8,
                        kunlun::packed_i8_scales(di8.data(),w.k,w.n),kunlun::packed_i8_maxima(di8.data(),w.k,w.n));
                    kunlun::check(kunlun::wait());
                } else require(kunlun::native_q4_matmul(native,pd,dy.data(),dw.data(),dx.data(),w.k,w.n));
                kunlun::check(kunlun::copy(y.data(),dy.data(),y.size()*4,kunlun::device_to_host));kunlun::check(kunlun::wait());
                auto e=error(gold,y);require(e.nmse<(cdnn?.01:1e-5));
                auto s=native.statistics();
                std::printf("MIXED PASS pd=%d legacy_first=%d n=%d step=%d mode=%s nmse=%.9g hbm_workspace=%zu l3_workspace=%zu\n",
                    pd,legacy_first,w.n,step,cdnn?"cdnn":"native",e.nmse,s.hbm_workspace_bytes,s.l3_workspace_bytes);
            }
        }
    }
}
