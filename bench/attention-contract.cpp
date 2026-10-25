#include <kunlun/k200.h>
#include <climits>
#include <cstdint>
#include <cstdio>
using namespace kunlun;
static Tensor4Layout dense(int x,int y,int h,InputType t){int b=t==InputType::f16?2:4;return {{x,y,h,1},{b,x*b,x*y*b,x*y*h*b},t};}
int main(){
 Kernel kernel;Matmul4Layout base{dense(64,16,4,InputType::f16),dense(64,3,8,InputType::f32),dense(16,3,8,InputType::f32),true};
 if(*attention_matmul_reason(base))return 1;
 for(int test=0;test<13;++test){auto p=base;int id=0;void *y=(void*)0x100000;const void *a=(void*)0x200000,*b=(void*)0x300000;size_t yb=1536,ab=8192,bb=6144;
  switch(test){case 0:p.a.ne[0]=0;break;case 1:p.b.ne[1]=257;break;case 2:p.a.ne[2]=3;break;case 3:p.a.nb[0]=4;break;case 4:p.y.type=InputType::f16;break;case 5:yb=4;break;case 6:a=nullptr;break;case 7:y=(void*)0x100001;break;case 8:y=(void*)((uintptr_t)a+128);break;case 9:id=2;break;case 10:p.a.ne[1]=65536;p.a.nb[1]=INT_MAX-1;break;case 11:y=(void*)(UINTPTR_MAX-8);break;case 12:p.a.type=(InputType)99;break;}
  if(attention_matmul(kernel,id,y,yb,a,ab,b,bb,p))return 1;std::printf("ATTENTION_CONTRACT matrix=%d rejected_before_device=1 PASS\n",test);
 }
 for(int test=0;test<7;++test){auto x=dense(257,3,2,InputType::f32);Softmax4Layout p{x,x,dense(257,3,1,InputType::f16),true,1,0};void*y=(void*)0x100000;const void *src=(void*)0x200000,*mask=(void*)0x300000;size_t bytes=257*3*2*4;
  switch(test){case 0:p.mask.ne[0]=256;break;case 1:p.max_bias=-1;break;case 2:p.x.type=InputType::f16;break;case 3:y=(void*)0x100001;break;case 4:mask=nullptr;break;case 5:y=(void*)((uintptr_t)src+4);break;case 6:p.y.nb[1]=4;break;}
  if(attention_softmax(kernel,0,y,bytes,src,bytes,mask,257*3*2,p))return 1;std::printf("ATTENTION_CONTRACT softmax=%d rejected_before_device=1 PASS\n",test);
 }
 for(int test=0;test<5;++test){auto x=dense(64,4,3,InputType::f32);Rope4Layout p{x,x,64,0,0,10000,1,1};const int32_t*pos=(const int32_t*)0x300000;size_t pb=12;
  switch(test){case 0:p.n_dims=63;break;case 1:p.mode=8;break;case 2:p.freq_base=0;break;case 3:pb=0;break;case 4:pos=nullptr;break;}
  if(attention_rope(kernel,0,(void*)0x100000,3072,(void*)0x200000,3072,pos,pb,p))return 1;std::printf("ATTENTION_CONTRACT rope=%d rejected_before_device=1 PASS\n",test);
 }
 if(allocated_bytes(0)||allocated_bytes(1))return 1;
}
