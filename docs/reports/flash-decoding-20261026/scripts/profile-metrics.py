from pathlib import Path
import json,re,statistics
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';P=json.loads((O/'profile-summary.json').read_text());out=[]
for name,data in P.items():
 for mode in ['decode','legacy']:
  records=[r for r in data if ('Flash-Decoding' in r.get('banner',''))==(mode=='decode')]
  if not records:continue
  r=records[-1];row=dict(test=name,mode=mode,banner=r['banner'],stages={})
  for st in ['kq','online','pv','reduce','final']:
   a=[w for w in r['workers'] if w['stage']==st]
   if not a:continue
   if a[0]['kind']=='sd':
    groups=[a[i:i+4] for i in range(0,len(a),4)]
    dd={f't{i}':sum(int(w[f't{i}']) for w in a) for i in range(16)}
    dd['critical_ticks']=sum(max(int(w['t9']) for w in g) for g in groups);dd['launches']=len(groups);dd['active_sdnn']=[sum(int(w['t14'])>0 for w in g) for g in groups]
    dd['critical_phase_ticks']=[sum(max(int(w[f't{i}']) for w in g) for g in groups) for i in range(9)]
   else:
    groups=[a[i:i+64] for i in range(0,len(a),64)]
    dd=dict(launches=len(groups),active_cl=[sum(int(w['rows'])>0 for w in g) for g in groups],rows=sum(int(w['rows']) for w in a),elements=sum(int(w['elems']) for w in a),critical_ticks=sum(max(int(w['ticks']) for w in g) for g in groups))
   row['stages'][st]=dd
  out.append(row)
(O/'worker-comparison.json').write_text(json.dumps(out,indent=2))
for r in out:
 if r['test'].startswith('profile-p0') and r['test'].endswith('c4'):
  print(r['test'],r['mode'])
  for k,v in r['stages'].items():print(k,v)
