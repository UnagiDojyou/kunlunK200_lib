#pragma once
#include "../q4-device.h"
#include "xpu/kernel/simd.h"
enum { NEXT_CHUNK=64, NEXT_PEAKS=288 };
static inline __device__ float next_float(unsigned int u){union{unsigned int u;float f;}v;v.u=u;return v.f;}
static inline __device__ unsigned int next_bits(float f){union{unsigned int u;float f;}v;v.f=f;return v.u;}
static inline __device__ float next_max(const unsigned int *v,int count){
 unsigned int m=0;for(int i=0;i<count;++i)if(v[i]>m)m=v[i];return m?next_float(m):1.f;
}
static inline __device__ int next_round(float v){int q=(int)(v+(v>=0?.5f:-.5f));if(q>127)q=127;if(q<-127)q=-127;return q;}
