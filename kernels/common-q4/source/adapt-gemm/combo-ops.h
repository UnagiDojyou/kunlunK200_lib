#pragma once
#define q40_finish big_original_finish
#include "big-ops.h"
#undef q40_finish
static inline __device__ void q40_finish(int rows,int nt,int acc,int scale,int blocks){
    int plane=rows*(nt/16);v16f32* e=(v16f32*)0;
    ew_cfg_activation_type(EW_NOACT);ew_cfg_findmax(0,0.f);ew_cfg_col_size(16);ew_cfg_coeff_scalar(1.f,1.f,0.f);
    for(int live=blocks;live>1;live/=2){ew_cfg_stream_size(live/2*plane);ew_dsmadd(&e[acc],&e[acc],&e[acc+live/2*plane],EW_L1E);xfence();}
    ew_cfg_stream_size(plane);ew_cfg_coeff_scalar(1.f,0.f,0.f);ew_dsmadd((v16f32*)0,&e[acc],&e[acc],EW_L2E);
}
