from pathlib import Path
import os,subprocess,sys
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-online-20261026'
subprocess.run(['g++','-std=c++17','-O3','-I'+str(L/'include'),'-I'+str(R/'external/xre/include'),'-I'+str(L/'kernels'),str(O/'online-bench.cpp'),'-L'+str(R/'build/flash-online-lib/lib'),'-lkunlunk200','-L'+str(R/'external/xre/lib64'),'-lxpurt','-Wl,-rpath,'+str(R/'build/flash-online-lib/lib'),'-o',str(O/'online-bench')],check=True)
sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['build/flash-online-lib/lib','external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for pd in [0,1]:
 for m,block in [(128,512),(1,1024)]:
  for mask in [0,1,2,4]:
   text=runner.run(f'ordered-online-p{pd}-m{m}-mask{mask}',[O/'online-bench',pd,128,block,m,32,mask,1,R/'build/flash-online-lib/lib/kunlunk200/kernels'],seconds=60)
   print('\n'.join(l for l in text.splitlines() if l.startswith('ONLINE_UNIT')),flush=True)
  for mask in [0,1,2]:
   for mode in ['legacy','ordered']:
    text=runner.run(f'profile-p{pd}-m{m}-mask{mask}-{mode}',[O/'online-bench',pd,128,block,m,32,mask,1,O/'profile','-profile-'+mode],seconds=60)
    print('\n'.join(l for l in text.splitlines() if l.startswith(('ONLINE_UNIT','ONLINE_STAGE'))),flush=True)
 for mask in [6,7]:
  for phase in [0,1]:
   text=runner.run(f'special-online-p{pd}-mask{mask}-phase{phase}',[O/'online-bench',pd,128,512,128,32,mask,phase,R/'build/flash-online-lib/lib/kunlunk200/kernels'],seconds=60)
   print('\n'.join(l for l in text.splitlines() if l.startswith('ONLINE_UNIT')),flush=True)
print('PROFILE COMPLETE',flush=True)
