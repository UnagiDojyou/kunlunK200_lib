from pathlib import Path
import re,subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-kq-20261026';P=O/'profile';P.mkdir(exist_ok=True)
env=os.environ.copy();env['XTDK_DIR']=str(R/'external/xtdk')
for kind in ['legacy','reuse32','reuse1024']:
 old=kind=='legacy';s=(L/('kernels/kunlun-flash-kq.xpu' if old else 'kernels/flash-kq-device.h')).read_text()
 s=s.replace('kunlun_flash_kq(',f'kunlun_flash_kq_profile_{kind}(').replace('KQ_NAME(',f'kunlun_flash_kq_profile_{kind}(')
 if not old:s=f'#define KQ_OUTPUT_KEYS {32 if kind=="reuse32" else 1024}\n'+s
 s=s.replace('"flash-config.h"','"'+str(L/'kernels/flash-config.h')+'"')
 s=s.replace('int port,float peak){','int port,float peak,unsigned int *t){').replace(',1,a[0]);',',1,a[0],t);').replace(',0,r[0]);',',0,r[0],t);')
 s=s.replace('int s0,int count){','int s0,int count,unsigned int *profile){')
 s=s.replace(' if(core_id()!=0)return;',' if(core_id()!=0)return;\n __local__ unsigned int t[16];for(int j=0;j<16;++j)t[j]=0;unsigned int kernel_begin=get_clock();')
 def wrap(engine,stage,count):
  nonlocal_dummy=None
  pattern=r'xfence_lock\('+re.escape(engine)+r'\);.*?xfence_unlock\('+re.escape(engine)+r'\);'
  def replace(m):return '{unsigned int phase_begin=get_clock();'+m[0]+f't[{stage}]+=(unsigned int)get_clock()-phase_begin;{count}'+'}'
  return re.sub(pattern,replace,s,flags=re.S)
 s=wrap('engine','port?2:0','t[port?11:10]++;')
 s=wrap('port?DS_1:DS_0','port?3:1','t[port?13:12]+=(rows+15)/16;')
 s=wrap('DMAIN_0','0','t[10]++;')
 s=wrap('DS_0','1','t[12]+=(groups+15)/16;')
 if not old:
  s=wrap('DMAIN_1','2','t[11]++;');s=wrap('DS_1','3','t[13]+=kk/16;')
 s=wrap('MAC','4','t[14]+='+('2;' if old else 'vectors;'))
 s=wrap('EW','5','');s=wrap('RS','6','');s=wrap('DMAOUT','7','t[15]++;')
 pos=s.rfind('}')
 s=s[:pos]+'t[9]=(unsigned int)get_clock()-kernel_begin;unsigned int used=0;for(int j=0;j<8;++j)used+=t[j];t[8]=t[9]>used?t[9]-used:0;LM2GM(t,profile+cluster_id()*16,64);\n'+s[pos:]
 f=P/f'kunlun-flash-kq-profile-{kind}.xpu';f.write_text(s)
 subprocess.run(['bash',str(L/'cmake/compile-native.sh'),str(f),str(P)],env=env,check=True)
for name in ['kunlun-flash-kq','kunlun-flash-kq-reuse32','kunlun-flash-kq-reuse1024']:
 f=R/'build/flash-kq-lib/lib/kunlunk200/kernels'/(name+'.device.s');(O/f.name).write_text(f.read_text());print(name,'frame',re.findall(r'addi sp, sp, (-\d+)',f.read_text()))
