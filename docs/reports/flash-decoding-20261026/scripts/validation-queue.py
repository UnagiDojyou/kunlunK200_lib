from pathlib import Path
import time,json,runpy
O=Path('/home/unagidojyou/kunlun-k200/results/flash-decoding-20261026')
deadline=time.monotonic()+3600
while True:
 if (O/'XPU_ABORT').exists():raise RuntimeError('stop latch')
 p=O/'logs/long-perf-7b-on.json'
 if p.exists():
  assert json.loads(p.read_text())['returncode']==0
  break
 if time.monotonic()>deadline:raise RuntimeError('performance queue did not finish')
 time.sleep(5)
for name in ['extra.py','validate.py']:
 print('BEGIN',name,flush=True);runpy.run_path(str(O/name),run_name='__main__');print('END',name,flush=True)
(O/'VALIDATION_DONE').write_text('complete\n')
