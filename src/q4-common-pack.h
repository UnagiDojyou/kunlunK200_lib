#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace common_q4 {
enum class Layout:uint32_t {GemvDual16=0,Micro32Bank=1,Panel512Bank=2};
enum class ScaleOrder:uint32_t {N32Major=0,KMajor=1};
struct Info {
    uint32_t abi_version=1,quant_type=2;Layout layout;ScaleOrder scale_order;
    int k,n,kp,np;size_t weight_offset=0,scale_offset,payload_bytes;uint32_t alignment=256;
};
inline int up(int n,int a){return (n+a-1)/a*a;}
inline Info describe(int k,int n,Layout layout,ScaleOrder scale){
    if(k<1||k>16384||n<1||n>32000||uint32_t(layout)>2||uint32_t(scale)>1)throw std::runtime_error("common Q4 unsupported dimensions/layout");
    Info a{};a.k=k;a.n=n;a.kp=up(k,k<512?512:256);a.np=up(n,256);a.layout=layout;a.scale_order=scale;
    a.scale_offset=size_t(a.kp)*a.np/2;a.payload_bytes=a.scale_offset+size_t(a.kp)*a.np/16;return a;
}
// A byte contains adjacent K values. row = hiN16*8 + (pairK%16)/2;
// bank = N%16. These are word coordinates, not a numerical conversion.
inline size_t weight_index(const Info& a,int col,int pair){
    int kb=pair/16,p=pair%16,row=(col%32/16)*8+p/2,bank=col%16;
    size_t tile=size_t(col/32)*(a.kp/32);
    if(a.layout==Layout::GemvDual16)return (tile+kb)*512+(row*16+bank)*2+p%2;
    if(a.layout==Layout::Micro32Bank)return (tile+kb)*512+(bank*16+row)*2+p%2;
    int base=kb/16*16,count=std::min(16,a.kp/32-base);
    return (tile+base)*512+bank*(count*32)+(kb-base)*32+row*2+p%2;
}
inline size_t scale_index(const Info& a,int col,int kb){return a.scale_offset+(a.scale_order==ScaleOrder::N32Major?(size_t(col/32)*(a.kp/32)+kb)*32+col%32:size_t(kb)*a.np+col)*2;}
struct Pack {
    Info info;std::vector<unsigned char> bytes;double pack_ms=0;
    Pack(const unsigned char* raw,size_t length,int k,int n,Layout layout,ScaleOrder scale,bool verify=true):info(describe(k,n,layout,scale)),bytes(info.payload_bytes,0){
        auto start=std::chrono::steady_clock::now();int blocks=(k+31)/32;
        if(!raw||length!=size_t(n)*blocks*18)throw std::runtime_error("original Q4 block size");
        for(int c=0;c<n;++c)for(int b=0;b<blocks;++b){
            const auto* src=raw+(size_t(c)*blocks+b)*18;size_t si=scale_index(info,c,b);bytes[si]=src[0];bytes[si+1]=src[1];
            auto q=[&](int j){return int((src[2+j%16]>>(j/16*4))&15)-8;};
            for(int p=0;p<16;++p)bytes[weight_index(info,c,b*16+p)]=(q(p*2)&15)|((q(p*2+1)&15)<<4);
        }
        pack_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(verify && inverse()!=std::vector<unsigned char>(raw,raw+length))throw std::runtime_error("common Q4 inverse mismatch");
    }
    int q(int c,int j)const {int v=bytes[weight_index(info,c,j/2)]>>(j%2*4)&15;return v>=8?v-16:v;}
    unsigned short scale_bits(int c,int b)const{size_t i=scale_index(info,c,b);return bytes[i]|(bytes[i+1]<<8);}
    std::vector<unsigned char> inverse()const {
        int blocks=(info.k+31)/32;std::vector<unsigned char> raw(size_t(info.n)*blocks*18);
        for(int c=0;c<info.n;++c)for(int b=0;b<blocks;++b){size_t o=(size_t(c)*blocks+b)*18;unsigned short h=scale_bits(c,b);raw[o]=h&255;raw[o+1]=h>>8;
            for(int j=0;j<16;++j)raw[o+2+j]=(q(c,b*32+j)+8)|((q(c,b*32+j+16)+8)<<4);
        }return raw;
    }
};
} // namespace common_q4
