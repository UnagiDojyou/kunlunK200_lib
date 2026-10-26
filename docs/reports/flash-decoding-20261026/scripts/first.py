from pathlib import Path
import os,sys
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';sys.path.insert(0,str(O));import runner;runner.OUT=O
os.environ['LD_LIBRARY_PATH']=':'.join(str(R/x) for x in ['external/xre/lib64','external/xdnn/so','external/xtdk/shlib'])
for d,hk in [(64,4),(128,32)]:
 for c in [1,2,4,8]:
  t=runner.run(f'first-d{d}-c{c}',[R/'build/flash-decode-lib/k200-flash-decode-bench',0,d,2048,1,hk,32,12],{'KUNLUN_FLASH_DECODING':'on','KUNLUN_FLASH_DECODING_CHUNKS':str(c),'KUNLUN_FLASH_KQ':'auto','KUNLUN_FLASH_ONLINE':'auto'},120)
  print('\n'.join(l for l in t.splitlines() if l.startswith(('KUNLUN Flash-Decoding','DECODE_COMPARE','FLASH_UNIT'))),flush=True)
