from pathlib import Path
import os,sys,subprocess
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-online-20261026'
subprocess.run(['g++','-std=c++17','-O3','-I'+str(L/'include'),'-I'+str(R/'external/xre/include'),'-I'+str(L/'kernels'),str(O/'online-bench.cpp'),'-L'+str(R/'build/flash-online-lib/lib'),'-lkunlunk200','-L'+str(R/'external/xre/lib64'),'-lxpurt','-Wl,-rpath,'+str(R/'build/flash-online-lib/lib'),'-o',str(O/'online-bench')],check=True)
sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['build/flash-online-lib/lib','external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for m,block in [(128,512),(1,1024)]:
 for phase,kind in [(1,'split'),(2,'profile')]:
  for mode in ['legacy','ordered']:
   text=runner.run(f'extra-{kind}-m{m}-phase{phase}-{mode}',[O/'online-bench',0,128,block,m,32,1,phase,O/'profile','-'+kind+'-'+mode],seconds=60)
   print('\n'.join(l for l in text.splitlines() if l.startswith(('ONLINE_UNIT','ONLINE_STAGE'))),flush=True)
print('EXTRA PROFILE COMPLETE',flush=True)
