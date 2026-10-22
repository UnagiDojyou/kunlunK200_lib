#pragma once
#if GEMM_PROFILE
#define DETAIL(slot,code) {unsigned int stamp=(unsigned int)get_clock();code;dt[slot]+=(unsigned int)get_clock()-stamp;}
#define COUNT(slot) {dt[slot]++;}
#define ISSUE_BEGIN unsigned int issue_start=(unsigned int)get_clock();
#define ISSUE_END(cid) {if(cid!=5)dt[cid==2?17:cid==1?20:cid==4?22:30]+=(unsigned int)get_clock()-issue_start;if(cid==2)dt[27]+=blocks*(nt/32);}
#else
#define DETAIL(slot,code) {code;}
#define COUNT(slot) {}
#define ISSUE_BEGIN
#define ISSUE_END(cid)
#endif
static inline __device__ int wait_slot(int cid){return cid==5?15:cid==2?18:cid==1?21:cid==4?23:cid==0?28:29;}
