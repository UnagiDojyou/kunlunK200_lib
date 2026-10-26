namespace {
void flash_profile_report(const char *stage,const FlashState &s,const void *stats=nullptr){
 bool kq=!std::strcmp(stage,"kq"),pv=!std::strcmp(stage,"pv"),online=!std::strcmp(stage,"online")||!std::strcmp(stage,"final"),reduce=!std::strcmp(stage,"reduce");
 if(!kq&&!pv&&!online&&!reduce)return;
 std::vector<unsigned int> data(256);const void *src=(kq||pv)?s.peaks->data():(reduce?stats:s.ranges->data());if(!src)return;
 check(xpu_memcpy(data.data(),src,1024,XPU_DEVICE_TO_HOST));check(xpu_wait());
 if(kq||pv){int base=pv?192:128;for(int cl=0;cl<4;++cl){std::fprintf(stderr,"FLASH_WORK stage=%s kind=sd cluster=%d",stage,cl);for(int j=0;j<16;++j)std::fprintf(stderr," t%d=%u",j,data[base+cl*16+j]);std::fprintf(stderr,"\n");}}
 else for(int core=0;core<64;++core){unsigned int m[3];for(int j=0;j<3;++j){int slot=core*3+j;m[j]=reduce?data[slot/6*8+slot%6+2]:data[slot/7*8+slot%7+1];}std::fprintf(stderr,"FLASH_WORK stage=%s kind=cl cluster=%d core=%d ticks=%u rows=%u elems=%u\n",stage,core/16,core%16,m[0],m[1],m[2]);}
}
}
