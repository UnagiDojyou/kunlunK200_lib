from pathlib import Path
import subprocess
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026'
print((O/'XPU_ABORT').read_text())
print((O/'logs/profile-p0-hk1-c1.txt').read_text()[-7000:])
subprocess.run(['sudo','-n','python3',str(O/'check_users.py')],check=True)
for pd in [0,1]:
 for k in ['state','errinfo','errtask','sessions']:
  p=Path(f'/proc/xpu/dev{pd}/{k}');print(str(p),p.read_text() if p.exists() else 'absent')
print(subprocess.check_output([str(R/'external/xre/bin/xpu_smi'),'--help'],text=True,stderr=subprocess.STDOUT)[-5000:])
