from pathlib import Path
import os,sys
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-kq-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for s in [32,64,128]:
 for hk,h in [(1,8),(2,8),(4,32),(32,32)]:
  for m in [1,128]:
   name=f'small-s{s}-hk{hk}-m{m}';text=runner.run(name,[R/'build/flash-kq-lib/k200-flash-kq-api-bench',0,128,s,m,hk,h,12],{'KUNLUN_FLASH_KQ':'reuse1024','KUNLUN_FLASH_ONLINE':'auto','KUNLUN_FLASH_BENCH_SAMPLE':'1'},120)
   print('\n'.join(l for l in text.splitlines() if l.startswith(('FLASH_UNIT','KQ_COMPARE'))),flush=True)
