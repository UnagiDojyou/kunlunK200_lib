from pathlib import Path
import subprocess,os,json,time
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
for k in ['online','decode-online','decode-reduce']:
 p=L/f'kernels/kunlun-flash-profile-{k}.xpu';s=p.read_text().replace('__local__ unsigned int meta[3];','__local__ unsigned int meta[8],transfer[8];').replace('LM2GM(meta+j,','transfer[0]=meta[j];LM2GM(transfer,');p.write_text(s)
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
subprocess.run(['cmake','--build',str(R/'build/flash-decode-lib'),'-j6','--target','k200-flash-decode-bench'],env=env,check=True)
subprocess.run(['sudo','-n','python3',str(O/'check_users.py')],check=True)
print(subprocess.check_output(['ps','-p','1153924','-o','pid,stat,cmd'],text=True))
print('ready for recovery')
