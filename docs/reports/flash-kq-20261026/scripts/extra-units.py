from pathlib import Path
import os,sys,json
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
K=R/'build/flash-kq-lib/lib/kunlunk200/kernels'
for pd in [0,1]:
 for d,hk in [(64,4),(128,32)]:
  for s in [256,512,2048,4096]:
   for m in [1,128]:
    text=runner.run(f'kq-grid-p{pd}-d{d}-s{s}-m{m}',[O/'kq-bench',pd,d,s,m,hk,32,K],{},120);print('\n'.join(l for l in text.splitlines() if l.startswith('KQ_UNIT')),flush=True)
 for hk,h,pat,batch in [(1,8,0,1),(2,8,3,1),(2,8,11,1),(2,8,18,4),(8,32,4,1),(8,32,8,1),(8,32,9,1),(8,32,17,1)]:
  m=1 if pat==17 else 3
  text=runner.run(f'kq-edge-p{pd}-hk{hk}-pat{pat}',[O/'kq-bench',pd,128,544,m,hk,h,K,pat,batch],{},120);print('\n'.join(l for l in text.splitlines() if l.startswith('KQ_UNIT')),flush=True)
 for qt in [16,32,64,128]:
  text=runner.run(f'kq-qt-p{pd}-qt{qt}',[O/'kq-bench',pd,128,512,128,8,32,K,0,1,qt],{},120);print('\n'.join(l for l in text.splitlines() if l.startswith('KQ_UNIT')),flush=True)
 for m,hk,d in [(128,4,64),(1,4,64),(128,32,128),(1,32,128)]:
  for mode in ['legacy','reuse32','reuse1024']:
   text=runner.run(f'profile-p{pd}-d{d}-m{m}-{mode}',[O/'kq-bench',pd,d,2048,m,hk,32,O/'profile',0,1,128,'-profile-'+mode],{},120);print('\n'.join(l for l in text.splitlines() if l.startswith(('KQ_UNIT','KQ_STAGE'))),flush=True)
print('EXTRA KQ UNITS COMPLETE',flush=True)
