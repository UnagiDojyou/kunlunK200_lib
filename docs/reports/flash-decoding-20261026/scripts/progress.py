from pathlib import Path
import re,statistics,json
O=Path('/home/unagidojyou/kunlun-k200/results/flash-decoding-20261026')
for p in sorted((O/'logs').glob('*perf-*.json')):
 if not p.stem.startswith(('perf-','long-perf-','repeat-')):continue
 d=json.loads(p.read_text());s=p.with_suffix('.txt').read_text(errors='replace');groups={}
 for l in s.splitlines():
  if not l.startswith('K200PERF '):continue
  x=dict(re.findall(r'(\w+)=([^ ]+)',l));n=int(x['pp_tokens']);groups.setdefault(n,[]).append(float(x['tg_steps'])*1000/float(x['tg_ms']))
 print(p.stem,{n:round(statistics.median(v),4) for n,v in groups.items()})
print('validation_done',(O/'VALIDATION_DONE').exists(),'abort',(O/'XPU_ABORT').exists())
