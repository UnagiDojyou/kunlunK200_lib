#include <kunlun/k200.h>
#include <climits>
#include <cstdio>

int main() {
    kunlun::Kernel k;
    kunlun::SwiGLULayout base{256,2,1024,1024,1024,kunlun::InputType::f32};
    if(*kunlun::swiglu_reason(base))return 1;
    for(int test=0;test<11;++test) {
        auto p=base;int pd=0;void *y=(void*)0x100000;
        const void *g=(void*)0x200000,*u=(void*)0x300000;size_t yb=2048,gb=2048,ub=2048;
        switch(test) {
            case 0:p.width=0;break;
            case 1:p.rows=0;break;
            case 2:p.width=65537;break;
            case 3:p.gate_stride=1023;break;
            case 4:p.rows=INT_MAX;break;
            case 5:p.type=(kunlun::InputType)99;break;
            case 6:yb=1024;break;
            case 7:g=nullptr;break;
            case 8:y=(void*)0x100001;break;
            case 9:y=(char*)g+4;break;
            case 10:pd=-1;break;
        }
        if(kunlun::swiglu(k,pd,y,yb,g,gb,u,ub,p))return 1;
        std::printf("SWIGLU_CONTRACT test=%d rejected_before_device=1 PASS\n",test);
    }
    if(kunlun::allocated_bytes(0) || kunlun::allocated_bytes(1))return 1;
}
