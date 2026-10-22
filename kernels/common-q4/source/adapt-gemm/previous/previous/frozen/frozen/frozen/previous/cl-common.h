#pragma once
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/simd.h"
static __device__ float from_bits(unsigned int u){union{unsigned int u;float f;}v;v.u=u;return v.f;}
static __device__ unsigned int bits(float f){union{unsigned int u;float f;}v;v.f=f;return v.u;}
static __device__ float half_value(unsigned int h){
    unsigned int e=(h>>10)&31,m=h&1023,s=(h&32768)<<16;
    if(!e){if(!m)return from_bits(s);int shift=0;while(!(m&1024)){m<<=1;++shift;}return from_bits(s|((113-shift)<<23)|((m&1023)<<13));}
    return from_bits(s|((e==31?255:e+112)<<23)|(m<<13));
}
static __device__ int round_quant(float v){int q=(int)(v+(v>=0?.5f:-.5f));if(q>127)q=127;if(q<-127)q=-127;return q;}
