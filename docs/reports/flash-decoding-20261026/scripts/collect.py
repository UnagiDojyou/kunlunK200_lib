from pathlib import Path
import json,re,statistics,csv,array,math,hashlib,subprocess
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
def fields(line):
 d={}
 for k,v in re.findall(r'(\w+)=([^\s]+)',line):
  try:d[k]=float(v) if any(c in v for c in '.e') else int(v)
  except ValueError:d[k]=v
 return d
def save(name,data):
 (O/(name+'.json')).write_text(json.dumps(data,indent=2))
 if isinstance(data,list) and data:
  with (O/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
units=[];trials=[];performance=[];stages=[]
for f in sorted((O/'logs').glob('*.txt')):
 meta=f.with_suffix('.json')
 if not meta.exists():continue
 meta=json.loads(meta.read_text());env=meta['environment'];name=f.stem;text=f.read_text(errors='replace')
 for line in text.splitlines():
  if line.startswith('FLASH_UNIT '):units.append(dict(test=name,**fields(line)))
 if name.startswith(('perf-','long-perf-','repeat-')):
  for flag in ['LD_PRELOAD','KUNLUN_FLASH_TRACE','KUNLUN_FLASH_DECODING_PROFILE','K200_CHECK_STATE','K200_TRACE_OPS']:assert flag not in env
  series='long' if name.startswith('long-perf') else ('repeat' if name.startswith('repeat') else 'main');model='tiny' if '-tiny-' in name else '7b';pd=int(meta['command'][-1]);group={}
  for line in text.splitlines():
   if not line.startswith('K200PERF '):continue
   row=dict(test=name,series=series,model=model,pd=pd,mode=env['KUNLUN_FLASH_DECODING'],**fields(line));trials.append(row);group.setdefault(row['pp_tokens'],[]).append(row)
  for n,data in group.items():
   pp=[x['pp_tokens']*1000/x['pp_ms'] for x in data];tg=[x['tg_steps']*1000/x['tg_ms'] for x in data]
   performance.append(dict(test=name,series=series,model=model,pd=pd,mode=env['KUNLUN_FLASH_DECODING'],context=n,pp_tps=statistics.median(pp),pp_min=min(pp),pp_max=max(pp),tg_tps=statistics.median(tg),tg_min=min(tg),tg_max=max(tg),trials=len(data)))
 if name.startswith('trace-'):
  mode='decode';i=0;api={}
  for line in text.splitlines():
   if line.startswith(('FLASH_STAGE ','DECODE_STAGE ')):
    d=fields(line);api[d['stage']]=api.get(d['stage'],0)+d['ms']
   elif line.startswith('KUNLUN Flash'):
    mode='decode' if 'Flash-Decoding' in line else 'legacy';stages.append(dict(test=name,mode=mode,iteration=i,**api));i+=1;api={}
comparison=[]
for x in performance:
 if x['mode']!='on':continue
 y=next((y for y in performance if y['mode']=='off' and all(y[k]==x[k] for k in ['series','model','pd','context'])),None)
 if y:comparison.append(dict(series=x['series'],model=x['model'],pd=x['pd'],context=x['context'],old_pp=y['pp_tps'],new_pp=x['pp_tps'],pp_gain_pct=100*(x['pp_tps']/y['pp_tps']-1),old_tg=y['tg_tps'],new_tg=x['tg_tps'],tg_gain_pct=100*(x['tg_tps']/y['tg_tps']-1)))
for n,data in [('units',units),('trials',trials),('performance',performance),('performance-comparison',comparison),('trace-stages',stages)]:save(n,data)
# Fixed teacher-forced continuation: same input IDs and prefix in both modes.
quality=[];teacher=list(map(int,(O/'inputs/teacher64.ids').read_text().split()))
for x in performance:
 if x['mode']!='on':continue
 y=next((y for y in performance if y['mode']=='off' and all(y[k]==x[k] for k in ['series','model','pd','context'])),None)
 if not y:continue
 n=x['context'];suffix='' if x['series']=='long' else '.p'+str(n)
 a=O/(y['test']+'.bin'+suffix);b=O/(x['test']+'.bin'+suffix)
 if not a.exists() or not b.exists():continue
 aa=array.array('f');aa.frombytes(a.read_bytes());bb=array.array('f');bb.frombytes(b.read_bytes());assert len(aa)==len(bb)
 nll=[0.,0.];maxabs=0.;sq=en=0.;kl=[];greedy=0
 for pos in range(len(aa)//32000):
  ar=aa[pos*32000:(pos+1)*32000];br=bb[pos*32000:(pos+1)*32000];assert all(map(math.isfinite,br));la=max(ar)+math.log(sum(math.exp(v-max(ar)) for v in ar)) if False else 0
  ma=max(ar);mb=max(br);la=ma+math.log(sum(math.exp(v-ma) for v in ar));lb=mb+math.log(sum(math.exp(v-mb) for v in br));nll[0]+=la-ar[teacher[pos]];nll[1]+=lb-br[teacher[pos]]
  kl.append(sum(math.exp(v-la)*(v-la-w+lb) for v,w in zip(ar,br)));greedy+=ar.index(ma)==br.index(mb)
  for v,w in zip(ar,br):maxabs=max(maxabs,abs(v-w));sq+=(v-w)**2;en+=v*v
 count=len(aa)//32000
 quality.append(dict(test=x['test'],context=n,model=x['model'],pd=x['pd'],positions=count,nmse=sq/max(en,1e-30),max_abs=maxabs,mean_KL=sum(kl)/count,max_KL=max(kl),greedy_equal=greedy,old_continuation_ppl=math.exp(nll[0]/count),new_continuation_ppl=math.exp(nll[1]/count),delta_nll=(nll[1]-nll[0])/count))
save('continuation-quality',quality)
summary=dict(unit_count=len(units),cpu_nmse_max=max((r['cpu_nmse'] for r in units),default=0),old_nmse_max=max((r['old_nmse'] for r in units),default=0),cpu_max_abs=max((r['cpu_max'] for r in units),default=0),old_max_abs=max((r['old_max'] for r in units),default=0),performance_comparison=comparison)
save('metrics',summary)
print('unit_count',len(units))
for r in comparison:print(r)
