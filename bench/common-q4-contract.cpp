#include <kunlun/k200.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc,char**argv){
    if(argc!=2)return 2;std::string test=argv[1];int k=512,n=512;
    if(test=="dimensions"){kunlun::CommonQ4Weight invalid(0,0,n);return 3;}
    if(test=="device"){kunlun::CommonQ4Weight invalid(2,k,n);return 3;}
    kunlun::Buffer x(0,k*4+256),y(0,n*4+256),storage(0,kunlun::common_q4_size(k,n)+256);
    if(test=="borrow-alignment"){kunlun::CommonQ4Weight invalid(0,k,n,(char*)storage.data()+4,storage.size()-4);return 3;}
    if(test=="borrow-capacity"){kunlun::CommonQ4Weight invalid(0,k,n,storage.data(),kunlun::common_q4_size(k,n)-1);return 3;}
    std::vector<unsigned char> raw((size_t)k*n/32*18,0x12);for(size_t i=0;i<raw.size();i+=18){raw[i]=0;raw[i+1]=0x3c;}
    if(test=="old-native-borrow")kunlun::upload_native_q4(0,storage.data(),raw.data(),k,n);
    kunlun::CommonQ4Weight weight(0,k,n,storage.data(),storage.size());
    if(test!="unready" && test!="old-native-borrow")weight.upload(raw.data(),raw.size());
    auto info=weight.info();info.abi_version=0;kunlun::require(weight.info().abi_version==1);
    kunlun::CommonQ4Workspace work;int pd=test=="foreign-pd"?1:0,m=test=="m-zero"?0:1;
    auto type=test=="type"?(kunlun::InputType)99:kunlun::InputType::f32;
    void* input=test=="input-alignment"?(void*)((char*)x.data()+1):x.data();
    void* output=test=="output-alignment"?(void*)((char*)y.data()+2):test=="overlap"?x.data():y.data();
    std::fprintf(stderr,"EXPECTED_CONTRACT_REJECTION case=%s\n",test.c_str());
    kunlun::common_q4_matmul(work,pd,output,weight,input,m,type);
    return 3;
}
