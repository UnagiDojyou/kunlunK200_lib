#pragma once
#ifndef GEMM_PROFILE
#define GEMM_PROFILE 0
#endif
#if GEMM_PROFILE
#define MEASURE(slot, code) {unsigned int ts=(unsigned int)get_clock();code;ticks[slot]+=(unsigned int)get_clock()-ts;}
#define JOIN() MEASURE(8,sync())
#else
#define MEASURE(slot, code) {code;}
#define JOIN() sync()
#endif
#define RUN_ENGINE(slot,engine,code) MEASURE(slot,{xfence_lock(engine);code;xfence_unlock(engine);})
