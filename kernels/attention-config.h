#pragma once
struct KunlunAttentionConfig {
    int dims[4], a[4], b[4], y[4], mask[4], mask_ne[4];
    int type_a, type_b, heads_a, batches_a, k, n, m, round_b;
    float scale, max_bias, theta_scale, freq_scale, attn_factor;
    int n_dims, mode, n_offs;
};


static_assert(sizeof(KunlunAttentionConfig)==160,"Attention configuration byte layout");
static_assert(__builtin_offsetof(KunlunAttentionConfig,type_a)==96,"Attention integer field offset");
static_assert(__builtin_offsetof(KunlunAttentionConfig,scale)==128,"Attention floating field offset");
