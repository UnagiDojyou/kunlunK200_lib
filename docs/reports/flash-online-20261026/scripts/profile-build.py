from pathlib import Path
import os,subprocess,re
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-online-20261026';P=O/'profile';P.mkdir(exist_ok=True)
env=os.environ.copy();env['XTDK_DIR']=str(R/'external/xtdk')
for kind in ['legacy','ordered']:
 s=(L/('kernels/kunlun-flash-online.xpu' if kind=='legacy' else 'kernels/flash-online-device.h')).read_text()
 s=s.replace('kunlun_flash_online(',f'kunlun_flash_online_profile_{kind}(').replace('ONLINE_NAME(',f'kunlun_flash_online_profile_{kind}(')
 s='#define ONLINE_REDUCE 2\n#define ONLINE_REUSE 1\n'+s
 for f in ['fp16-device.h','flash-config.h']:s=s.replace('"'+f+'"','"'+str(L/'kernels'/f)+'"')
 s=s.replace('  int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);','  int q=row%p.m,h=row/p.m%p.h,b=row/(p.m*p.h);\n  __local__ unsigned int timing[16];for(int i=0;i<16;++i)timing[i]=0;unsigned int begin=get_clock(),tick=begin,now;')
 def mark(i):return f'\n  now=get_clock();timing[{i}]=now-tick;tick=now;\n'
 s=s.replace('  if(count==0)',mark(0)+'  if(count==0)',1)
 s=s.replace('  if(mask){',mark(1)+'  if(mask){',1)
 s=s.replace('  float slope=1;',mark(2)+'  float slope=1;',1)
 s=s.replace('  for(int j=0;j<count;j+=8){_x256',mark(3)+'  for(int j=0;j<count;j+=8){_x256',1)
 if kind=='legacy':s=s.replace('  for(int j=0;j<count;++j)if(logits',mark(4)+'  for(int j=0;j<count;++j)if(logits',1)
 else:s=s.replace('#if ONLINE_REDUCE\n  float m0',mark(4)+'#if ONLINE_REDUCE\n  float m0',1)
 s=s.replace('  float alpha=',mark(5)+'  float alpha=',1)
 s=s.replace('  if(maximum==-__builtin_inff()){for',mark(6)+'  if(maximum==-__builtin_inff()){for',1)
 s=s.replace('  st[0]=maximum;',mark(7)+'  st[0]=maximum;',1)
 target='LM2GM(st,stats+row*8,32);'
 s=s.replace(target,target+mark(8)+'  timing[9]=get_clock()-begin;LM2GM(timing,y+row*64,64);',1)
 source=P/f'kunlun-flash-online-profile-{kind}.xpu';source.write_text(s)
 subprocess.run(['bash',str(L/'cmake/compile-native.sh'),str(source),str(P)],env=env,check=True)
 # Preserve production assembly, binary, and LM information independently.
for name in ['kunlun-flash-online','kunlun-flash-online-fast','kunlun-flash-online-reduce','kunlun-flash-online-reuse']:
 f=R/'build/flash-online-lib/lib/kunlunk200/kernels'/(name+'.device.s')
 asm=f.read_text();(O/(name+'.device.s')).write_text(asm)
 print('ISA',name,'frame',re.findall(r'addi sp, sp, (-\d+)',asm),'bytes',(f.with_suffix('.bin')).stat().st_size if f.with_suffix('.bin').exists() else 'see kernels')
