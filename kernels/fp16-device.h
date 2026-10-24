#pragma once
#include "xpu/kernel/cluster_header.h"
static __device__ float kunlun_from_half(unsigned int h){
    unsigned int e=(h>>10)&31,m=h&1023,sign=(h&32768)<<16;
    union{unsigned int u;float f;}value;
    if(e==0){
        if(m==0){value.u=sign;return value.f;}
        int shift=0;while((m&1024)==0){m<<=1;++shift;}
        value.u=sign|((113-shift)<<23)|((m&1023)<<13);
    }else value.u=sign|((e==31?255:e+112)<<23)|(m<<13);
    return value.f;
}

static __device__ unsigned short kunlun_to_half(float value) {
    union { float f; unsigned int u; } bits;
    bits.f=value;
    unsigned int sign=(bits.u>>16)&0x8000, mantissa=bits.u&0x7fffff;
    int exponent=((bits.u>>23)&255)-112;
    if (exponent==143) return sign|0x7c00|(mantissa ? ((mantissa>>13)|0x200) : 0);
    if (exponent>=31) return sign|0x7c00;
    if (exponent<=0) {
        if (exponent < -10) return sign;
        unsigned int m=mantissa|0x800000;
        int shift=14-exponent;
        return sign|((m+((1u<<(shift-1))-1)+((m>>shift)&1))>>shift);
    }
    mantissa+=0xfff+((mantissa>>13)&1);
    if (mantissa&0x800000) { mantissa=0; ++exponent; }
    if (exponent>=31) return sign|0x7c00;
    return sign|(exponent<<10)|(mantissa>>13);
}
