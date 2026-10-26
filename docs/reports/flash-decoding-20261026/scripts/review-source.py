from pathlib import Path
import subprocess
R=Path('/home/unagidojyou/kunlun-k200')
for p in [R/'repos/KunlunK200',R/'repos/llama.cpp']:
 print(p);print(subprocess.check_output(['git','status','--short'],cwd=p,text=True));print(subprocess.check_output(['git','diff','--stat'],cwd=p,text=True))
print((R/'repos/KunlunK200/src/flash-decode.inc').read_text()[:4200])
