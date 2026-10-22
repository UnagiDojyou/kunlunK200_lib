#pragma once
#include "cl-common.h"
enum {CHUNK=64,PEAKS=256};
static __device__ void read_activation(_global_ptr_ const void* input,int row,int base,int kp,int dtype,float* x,unsigned short* h){
    if(!dtype){GM2LM((_global_ptr_ const float*)input+(long long)row*kp+base,x,256);}
    else{GM2LM((_global_ptr_ const unsigned short*)input+(long long)row*kp+base,h,128);for(int j=0;j<64;++j)x[j]=half_value(h[j]);}
}
static __device__ float peak_value(unsigned int* peaks,int count){unsigned int mx=0;for(int j=0;j<count;++j)if(peaks[j]>mx)mx=peaks[j];return mx?from_bits(mx):1.f;}
