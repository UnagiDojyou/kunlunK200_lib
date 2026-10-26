from pathlib import Path
import subprocess,os
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-online-20261026';P=O/'profile'
env=os.environ.copy();env['XTDK_DIR']=str(R/'external/xtdk')
for mode in ['legacy','ordered']:
 source=P/f'kunlun-flash-online-profile-{mode}.xpu';s=source.read_text().replace(f'kunlun_flash_online_profile_{mode}(',f'kunlun_flash_online_split_{mode}(')
 s=s.replace('sum+=e0;sum+=e1;sum+=e2;sum+=e3;sum+=e4;sum+=e5;sum+=e6;sum+=e7;','')
 split='''for(int j=0;j<count;j+=8){sum+=logits[j];sum+=logits[j+1];sum+=logits[j+2];sum+=logits[j+3];sum+=logits[j+4];sum+=logits[j+5];sum+=logits[j+6];sum+=logits[j+7];}
  now=get_clock();timing[10]=now-tick;tick=now;
'''
 s=s.replace('  st[0]=maximum;',split+'  st[0]=maximum;',1)
 dest=P/f'kunlun-flash-online-split-{mode}.xpu';dest.write_text(s)
 subprocess.run(['bash',str(L/'cmake/compile-native.sh'),str(dest),str(P)],env=env,check=True)
