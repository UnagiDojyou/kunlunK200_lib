from pathlib import Path
import re,subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200'
names=[]
for kind,file,entry,is_pv in [('kq','flash-kq-device.h','KQ_NAME',False),('pv','kunlun-flash-pv-wide.xpu','kunlun_flash_pv_wide',True),('decode-kq','kunlun-flash-decode-kq.xpu','kunlun_flash_decode_kq',False),('decode-pv','kunlun-flash-decode-pv-head.xpu','kunlun_flash_decode_pv_head',True)]:
 name='kunlun-flash-profile-'+kind;s=(L/'kernels'/file).read_text().replace(entry,'kunlun_flash_profile_'+kind.replace('-','_'))
 if kind=='kq':s='#define KQ_OUTPUT_KEYS 1024\n'+s
 s=s.replace('int port,float peak){','int port,float peak,unsigned int *t){').replace(',0,r[0]);',',0,r[0],t);')
 s=s.replace('if(core_id()!=0)return;','if(core_id()!=0)return;\n __local__ unsigned int t[16];for(int i=0;i<16;++i)t[i]=0;unsigned int kernel_begin=get_clock();')
 def wrap(engine,stage,count):
  pattern=r'xfence_lock\('+re.escape(engine)+r'\);.*?xfence_unlock\('+re.escape(engine)+r'\);'
  return re.sub(pattern,lambda m:'{unsigned int phase_begin=get_clock();'+m[0]+f't[{stage}]+=(unsigned int)get_clock()-phase_begin;{count}'+'}',s,flags=re.S)
 if not is_pv:
  s=wrap('engine','port?2:0','t[port?11:10]++;');s=wrap('port?DS_1:DS_0','port?3:1','t[port?13:12]+=(rows+15)/16;')
 s=wrap('DMAIN_0',0,'t[10]++;');s=wrap('DS_0',1,'t[12]+='+('(rows+15)/16;' if is_pv else '(groups+15)/16;'))
 s=wrap('DMAIN_1',2,'t[11]++;');s=wrap('DS_1',3,'t[13]+='+('p.d/16;' if is_pv else 'kk/16;'))
 s=wrap('MAC',4,'t[14]+='+('p.d/16;' if is_pv else 'vectors;'));s=wrap('EW',5,'');s=wrap('RS',6,'');s=wrap('DMAOUT',7,'t[15]++;')
 pos=s.rfind('}');s=s[:pos]+'''t[9]=(unsigned int)get_clock()-kernel_begin;unsigned int used=0;for(int i=0;i<8;++i)used+=t[i];t[8]=t[9]>used?t[9]-used:0;
 LM2GM(t,(_global_ptr_ unsigned int*)(peaks+128)+cluster_id()*16,64);
'''+s[pos:];(L/'kernels'/(name+'.xpu')).write_text(s);names.append(name)
for kind,file,entry in [('online','flash-online-device.h','ONLINE_NAME'),('decode-online','kunlun-flash-decode-online-head.xpu','kunlun_flash_decode_online_head'),('decode-reduce','kunlun-flash-decode-reduce-head.xpu','kunlun_flash_decode_reduce_head')]:
 name='kunlun-flash-profile-'+kind;s=(L/'kernels'/file).read_text().replace(entry,'kunlun_flash_profile_'+kind.replace('-','_'))
 if kind=='online':s='#define ONLINE_REDUCE 2\n#define ONLINE_REUSE 1\n'+s
 pos=s.index('{',s.index('__global__'))+1;s=s[:pos]+'\n __local__ unsigned int meta[3];meta[1]=meta[2]=0;unsigned int kernel_begin=get_clock();\n'+s[pos:]
 if kind=='online':s=s.replace('int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);','int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);meta[1]++;meta[2]+=count;')
 elif kind=='decode-online':s=s.replace('int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);','int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);meta[1]++;meta[2]+=count;')
 else:s=s.replace('for(int row=tid;row<rows;row+=workers){','for(int row=tid;row<rows;row+=workers){meta[1]++;meta[2]+=fd.chunks;')
 pos=s.rfind('}');target='stats' if kind=='decode-reduce' else 'qscale';padding=6 if kind=='decode-reduce' else 7;start=2 if padding==6 else 1
 s=s[:pos]+f'''meta[0]=(unsigned int)get_clock()-kernel_begin;int physical=cluster_id()*core_num()+core_id();
 for(int j=0;j<3;++j){{int slot=physical*3+j,index=slot/{padding}*8+slot%{padding}+{start};LM2GM(meta+j,(_global_ptr_ unsigned int*){target}+index,4);}}
'''+s[pos:];(L/'kernels'/(name+'.xpu')).write_text(s);names.append(name)
(L/'src/flash-profile.inc').write_text(r'''namespace {
void flash_profile_report(const char *stage,const FlashState &s,const void *stats=nullptr){
 bool kq=!std::strcmp(stage,"kq"),pv=!std::strcmp(stage,"pv"),online=!std::strcmp(stage,"online")||!std::strcmp(stage,"final"),reduce=!std::strcmp(stage,"reduce");
 if(!kq&&!pv&&!online&&!reduce)return;
 std::vector<unsigned int> data(256);const void *src=(kq||pv)?s.peaks->data():(reduce?stats:s.ranges->data());if(!src)return;
 check(xpu_memcpy(data.data(),src,1024,XPU_DEVICE_TO_HOST));check(xpu_wait());
 if(kq||pv){int base=pv?192:128;for(int cl=0;cl<4;++cl){std::fprintf(stderr,"FLASH_WORK stage=%s kind=sd cluster=%d",stage,cl);for(int j=0;j<16;++j)std::fprintf(stderr," t%d=%u",j,data[base+cl*16+j]);std::fprintf(stderr,"\n");}}
 else for(int core=0;core<64;++core){unsigned int m[3];for(int j=0;j<3;++j){int slot=core*3+j;m[j]=reduce?data[slot/6*8+slot%6+2]:data[slot/7*8+slot%7+1];}std::fprintf(stderr,"FLASH_WORK stage=%s kind=cl cluster=%d core=%d ticks=%u rows=%u elems=%u\n",stage,core/16,core%16,m[0],m[1],m[2]);}
}
}
''')
p=L/'src/flash.inc';s=p.read_text().replace('decode_reduce_head,qpack','decode_reduce_head,profile_kq,profile_pv,profile_online,profile_decode_kq,profile_decode_pv,profile_decode_online,profile_decode_reduce,qpack').replace('&decode_reduce_head,&qpack','&decode_reduce_head,&profile_kq,&profile_pv,&profile_online,&profile_decode_kq,&profile_decode_pv,&profile_decode_online,&profile_decode_reduce,&qpack')
s=s.replace('#include "flash-decode.inc"','#include "flash-profile.inc"\n#include "flash-decode.inc"')
s=s.replace('bool trace=std::getenv("KUNLUN_FLASH_TRACE")!=nullptr;','bool profile=std::getenv("KUNLUN_FLASH_DECODING_PROFILE")&&p.q.ne[1]==1&&p.q.ne[2]*p.q.ne[3]>=32;\n    bool trace=std::getenv("KUNLUN_FLASH_TRACE")!=nullptr||profile;')
s=s.replace('tick=now;}};','tick=now;if(profile)flash_profile_report(name,s);}};',1)
s=s.replace('auto *kq_program=kq_optimized?&s.kq1024:&s.kq;','if(profile&&!kq_optimized)profile=false;\n    auto *kq_program=profile?&s.profile_kq:(kq_optimized?&s.kq1024:&s.kq);')
s=s.replace('load_kernel(*kq_program,kq_optimized?', 'load_kernel(*kq_program,profile?"kunlun-flash-profile-kq":(kq_optimized?').replace('"kunlun-flash-kq",id,true);bool wide','"kunlun-flash-kq"),id,true);bool wide')
s=s.replace('auto &pv_program=wide?s.pv_wide:s.pv;','auto &pv_program=profile?s.profile_pv:(wide?s.pv_wide:s.pv);').replace('load_kernel(pv_program,wide?', 'load_kernel(pv_program,profile?"kunlun-flash-profile-pv":(wide?').replace('"kunlun-flash-pv",id,true);','"kunlun-flash-pv"),id,true);')
s=s.replace('auto &online=optimized?s.online_ordered:s.online;','auto &online=profile?s.profile_online:(optimized?s.online_ordered:s.online);').replace('load_kernel(online,optimized?', 'load_kernel(online,profile?"kunlun-flash-profile-online":(optimized?').replace('"kunlun-flash-online",id,false);','"kunlun-flash-online"),id,false);');p.write_text(s)
p=L/'src/flash-decode.inc';s=p.read_text().replace('bool head=!std::getenv', 'bool profile=std::getenv("KUNLUN_FLASH_DECODING_PROFILE")&&base.h*base.b>=32;\n bool head=!std::getenv')
s=s.replace('auto &online=head?', 'auto &online=profile?s.profile_decode_online:(head?').replace(':s.decode_online;',':s.decode_online);').replace('auto &pv=head?', 'auto &pv=profile?s.profile_decode_pv:(head?').replace(':s.decode_pv;',':s.decode_pv);').replace('auto &reduce=head?','auto &reduce=profile?s.profile_decode_reduce:(head?').replace(':s.decode_reduce;',':s.decode_reduce);')
s=s.replace('load_kernel(s.decode_kq,"kunlun-flash-decode-kq",id,true);','auto &kq_program=profile?s.profile_decode_kq:s.decode_kq;load_kernel(kq_program,profile?"kunlun-flash-profile-decode-kq":"kunlun-flash-decode-kq",id,true);')
s=s.replace('load_kernel(pv,head?', 'load_kernel(pv,profile?"kunlun-flash-profile-decode-pv":(head?').replace('"kunlun-flash-decode-pv",id,true);','"kunlun-flash-decode-pv"),id,true);')
s=s.replace('load_kernel(online,head?', 'load_kernel(online,profile?"kunlun-flash-profile-decode-online":(head?').replace('"kunlun-flash-decode-online",id,false);','"kunlun-flash-decode-online"),id,false);')
s=s.replace('load_kernel(reduce,head?', 'load_kernel(reduce,profile?"kunlun-flash-profile-decode-reduce":(head?').replace('"kunlun-flash-decode-reduce",id,false);','"kunlun-flash-decode-reduce"),id,false);')
s=s.replace('bool trace=std::getenv("KUNLUN_FLASH_TRACE");','bool trace=std::getenv("KUNLUN_FLASH_TRACE")||profile;').replace('tick=now;}};','tick=now;if(profile)flash_profile_report(name,s,stats);}};').replace('native_launch(s.decode_kq,','native_launch(kq_program,')
s=s.replace('bool head=!std::getenv','if(profile){}\n bool head=!std::getenv') # head profiles use the head-major contract only
s=s.replace('auto &online=profile?', 'if(profile)head=true;\n auto &online=profile?',1);p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('foreach(name kunlun-flash-decode-online-head','foreach(name '+' '.join(names)+' kunlun-flash-decode-online-head');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench'],env=env,check=True)
