#pragma once
#include "xpu/kernel/cluster_header.h"
static __device__ float q4_half(unsigned int h){
    unsigned int e=(h>>10)&31,m=h&1023,sign=(h&32768)<<16;
    union{unsigned int u;float f;}value;
    if(e==0){
        if(m==0){value.u=sign;return value.f;}
        int shift=0;while((m&1024)==0){m<<=1;++shift;}
        value.u=sign|((113-shift)<<23)|((m&1023)<<13);
    }else value.u=sign|((e==31?255:e+112)<<23)|(m<<13);
    return value.f;
}
static __device__ float q4_block_scale(const unsigned char *p){return q4_half((unsigned int)p[0]|((unsigned int)p[1]<<8));}
