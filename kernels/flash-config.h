#pragma once
struct KunlunFlashConfig {
    int d,s,m,h,hk,b,bk,bv,block,qt;
    int q[4],k[4],v[4],y[4],mask[4],mask_ne[4];
    int mask_type,has_mask,tg_group;
    float scale,max_bias;
};
