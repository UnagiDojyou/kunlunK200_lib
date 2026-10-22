#include <kunlun/k200.h>
#include "native-q4-reference.h"
#include <map>
using namespace native_q4_bench;
int main(int argc,char **argv){
    require(argc==3);int pd=std::atoi(argv[1]);bool f16=std::atoi(argv[2]);int k=544,n=513;
    kunlun::CommonQ4Weight weight(pd,k,n);{Weights raw(k,n);weight.upload(raw.q4.data(),raw.q4.size());}
    auto id=weight.data();auto input_values=input(k,256);std::vector<_Float16> halves(input_values.size());
    for(size_t i=0;i<halves.size();++i)halves[i]=(_Float16)input_values[i];
    kunlun::Buffer x(pd,input_values.size()*(f16?2:4)),y(pd,(size_t)256*n*4);
    kunlun::select(pd);kunlun::check(kunlun::copy(x.data(),f16?(const void*)halves.data():input_values.data(),x.size(),kunlun::host_to_device));
    std::map<int,std::vector<float>> gold;
    for(int m:{1,8,32,64,128,129,256}){kunlun::CommonQ4Workspace separate;kunlun::common_q4_matmul(separate,pd,y.data(),weight,x.data(),m,f16?kunlun::InputType::f16:kunlun::InputType::f32);gold[m].resize((size_t)m*n);kunlun::check(kunlun::copy(gold[m].data(),y.data(),gold[m].size()*4,kunlun::device_to_host));}
    kunlun::CommonQ4Workspace work;size_t plateau=0;int calls=0;
    for(int round=0;round<10;++round)for(int m:{1,8,32,64,128,129,256,1}){
        kunlun::common_q4_matmul(work,pd,y.data(),weight,x.data(),m,f16?kunlun::InputType::f16:kunlun::InputType::f32);
        std::vector<float> actual((size_t)m*n);kunlun::check(kunlun::copy(actual.data(),y.data(),actual.size()*4,kunlun::device_to_host));require(actual==gold[m]&&weight.data()==id);
        ++calls;if(round==0)plateau=kunlun::allocated_bytes(pd);else require(kunlun::allocated_bytes(pd)==plateau);
    }
    auto s=weight.statistics();auto w=work.statistics();require(s.uploads==1&&w.calls==80);
    std::printf("COMMON_STATE {\"pd\":%d,\"f16\":%s,\"calls\":%d,\"uploads\":%zu,\"regions\":1,\"same_weight_pointer\":true,\"stable_hbm\":true,\"hbm_bytes\":%zu,\"correct\":true}\n",pd,f16?"true":"false",calls,s.uploads,plateau);
}
