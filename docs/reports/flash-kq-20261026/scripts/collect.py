from pathlib import Path
import json,csv,re,statistics,hashlib,subprocess
R=Path('/home/unagidojyou/kunlun-k200');L=R/'repos/KunlunK200';O=R/'results/flash-kq-20261026'
def fields(line):
 d={}
 for k,v in re.findall(r'(\w+)=([^\s]+)',line):
  try:d[k]=float(v) if any(c in v for c in '.e') else int(v)
  except ValueError:d[k]=v
 return d
def save(name,data):
 (O/(name+'.json')).write_text(json.dumps(data,indent=2))
 if data:
  with (O/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for row in data for k in row)));w.writeheader();w.writerows(data)
units=[];kq=[];stages=[];trials=[];perf=[]
for f in sorted((O/'logs').glob('*.txt')):
 meta=f.with_suffix('.json');meta=json.loads(meta.read_text()) if meta.exists() else None
 for line in f.read_text(errors='replace').splitlines():
  if line.startswith('FLASH_UNIT ') and f.stem.startswith(('grid-','edge-','mutate-','small-','final-check-')):units.append(dict(test=f.stem,**fields(line)))
  if line.startswith('KQ_UNIT '):kq.append(dict(test=f.stem,**fields(line)))
  if line.startswith('KQ_STAGE ') and meta:
   pd,d,s,m,hk,h=map(int,meta['command'][1:7]);row=dict(test=f.stem,pd=pd,D=d,S=s,M=m,HK=hk,H=h,**fields(line))
   for j in range(10):row['ms'+str(j)]=row['t'+str(j)]/900000.
   stages.append(row)
 if not f.stem.startswith(('final-perf-','repeat-tiny-','repeat-7b-short-')) or not meta:continue
 series='main' if f.stem.startswith('final-perf-') else 'repeat'
 env=meta['environment'];mode=env['KUNLUN_FLASH_KQ'];model='tiny' if '-tiny-' in f.stem else '7b';pd=int(meta['command'][-1]);group={}
 for line in f.read_text(errors='replace').splitlines():
  if not line.startswith('K200PERF '):continue
  row=dict(series=series,model=model,pd=pd,mode=mode,test=f.stem,**fields(line));trials.append(row);group.setdefault(row['pp_tokens'],[]).append(row)
 for n,data in group.items():
  pp=[x['pp_tokens']*1000/x['pp_ms'] for x in data];tg=[x['tg_steps']*1000/x['tg_ms'] for x in data]
  perf.append(dict(series=series,model=model,pd=pd,mode=mode,context=n,pp_tps=statistics.median(pp),pp_min=min(pp),pp_max=max(pp),tg_tps=statistics.median(tg),tg_min=min(tg),tg_max=max(tg),trials=len(data)))
save('units',units);save('kq-units',kq);save('stages',stages);save('trials',trials);save('performance',perf)
comparison=[]
for x in perf:
 if x['mode']!='reuse1024':continue
 old=next((y for y in perf if y['mode']=='legacy' and all(y[k]==x[k] for k in ['series','model','pd','context'])),None)
 if not old:continue
 comparison.append(dict(series=x['series'],model=x['model'],pd=x['pd'],context=x['context'],legacy_pp=old['pp_tps'],new_pp=x['pp_tps'],pp_gain_pct=100*(x['pp_tps']/old['pp_tps']-1),legacy_tg=old['tg_tps'],new_tg=x['tg_tps'],tg_gain_pct=100*(x['tg_tps']/old['tg_tps']-1)))
save('performance-comparison',comparison)
normal_kq=[x for x in kq if 'profile' not in x['mode']]
summary={'api_unit_count':len(units),'api_old_nmse_max':max((x['old_nmse'] for x in units),default=0),'api_cpu_nmse_max':max((x['cpu_nmse'] for x in units),default=0),'api_cpu_max_abs':max((x['cpu_max'] for x in units),default=0),'kq_count':len(normal_kq),'kq_bitdiff_max':max((x['bits'] for x in normal_kq),default=0),'kq_cpu_nmse_max':max((x['cpu_nmse'] for x in normal_kq),default=0),'kq_cpu_max_abs':max((x['cpu_max'] for x in normal_kq),default=0),'comparisons':comparison}
D=R/'build/flash-kq-lib/lib/kunlunk200/kernels';base=json.loads((O/'flash-online-lib-images.json').read_text());same={};missing=[]
for n,sha in base.items():
 if (D/n).exists():same[n]=hashlib.sha256((D/n).read_bytes()).hexdigest()==sha
 else:missing.append(n)
assert set(missing)=={'kunlun-flash-online-fast.bin','kunlun-flash-online-reduce.bin','kunlun-flash-online-reuse.bin'} and len(same)==51 and all(same.values())
summary['baseline_product_kernels_unchanged']=same;summary['baseline_experimental_not_built']=missing
summary['states']={p:{k:Path(f'/proc/xpu/dev{p}/{k}').read_text().strip() for k in ['state','errinfo']} for p in [0,1]}
summary['frames']={n:{'frame':re.findall(r'addi sp, sp, (-\d+)',(D/(n+'.device.s')).read_text()),'bytes':(D/(n+'.bin')).stat().st_size,'sha256':hashlib.sha256((D/(n+'.bin')).read_bytes()).hexdigest()} for n in ['kunlun-flash-kq','kunlun-flash-kq-reuse1024']}
summary['source']={n:{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=p,text=True)} for n,p in [('library',L),('llama',R/'repos/llama.cpp')]}
(O/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
