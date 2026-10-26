from pathlib import Path
import json,re,csv,statistics,hashlib,subprocess
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-online-20261026';L=R/'repos/KunlunK200'
def fields(line):
 out={}
 for k,v in re.findall(r'([\w]+)=([^\s]+)',line):
  try:out[k]=float(v) if any(x in v for x in '.e') else int(v)
  except ValueError:out[k]=v
 return out
def save(name,rows):
 (O/(name+'.json')).write_text(json.dumps(rows,indent=2))
 if rows:
  with (O/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
units=[];online=[];stages=[];trials=[];performance=[]
for f in sorted((O/'logs').glob('*.txt')):
 text=f.read_text(errors='replace')
 for line in text.splitlines():
  if line.startswith('FLASH_UNIT ') and (f.stem.startswith(('grid-','edge-','mutate-real-','final-check-'))):units.append(dict(test=f.stem,**fields(line)))
  if line.startswith('ONLINE_UNIT ') and f.stem.startswith(('ordered-online-','special-online-','extra-')):online.append(dict(test=f.stem,**fields(line)))
  if line.startswith('ONLINE_STAGE '):stages.append(dict(test=f.stem,**fields(line)))
 if not f.stem.startswith('final-perf-'):continue
 meta=f.with_suffix('.json')
 if not meta.exists():continue
 meta=json.loads(meta.read_text());env=meta['environment'];mode=env['KUNLUN_FLASH_ONLINE'];model='tiny' if '-tiny-' in f.stem else '7b';pd=int(meta['command'][-1]);group={}
 for line in text.splitlines():
  if not line.startswith('K200PERF '):continue
  row=dict(test=f.stem,model=model,pd=pd,mode=mode,**fields(line));trials.append(row);group.setdefault(row['pp_tokens'],[]).append(row)
 for n,values in group.items():
  pp=[x['pp_tokens']*1000/x['pp_ms'] for x in values];tg=[x['tg_steps']*1000/x['tg_ms'] for x in values]
  performance.append(dict(model=model,pd=pd,mode=mode,context=n,pp_tps=statistics.median(pp),pp_min=min(pp),pp_max=max(pp),tg_tps=statistics.median(tg),tg_min=min(tg),tg_max=max(tg),trials=len(values)))
save('units',units);save('online',online);save('stages',stages);save('trials',trials);save('performance',performance)
comparison=[]
for row in performance:
 if row['mode']!='ordered':continue
 old=next(x for x in performance if all(x[k]==row[k] for k in ['model','pd','context']) and x['mode']=='legacy')
 comparison.append(dict(model=row['model'],pd=row['pd'],context=row['context'],legacy_pp=old['pp_tps'],new_pp=row['pp_tps'],pp_gain_pct=100*(row['pp_tps']/old['pp_tps']-1),legacy_tg=old['tg_tps'],new_tg=row['tg_tps'],tg_gain_pct=100*(row['tg_tps']/old['tg_tps']-1)))
save('performance-comparison',comparison)
summary={'unit_count':len(units),'cpu_nmse_max':max((x['cpu_nmse'] for x in units),default=0),'cpu_max_abs':max((x['cpu_max'] for x in units),default=0),'old_nmse_max':max((x['old_nmse'] for x in units),default=0),'old_max_abs':max((x['old_max'] for x in units),default=0),'comparisons':comparison}
summary['states']={pd:{k:Path(f'/proc/xpu/dev{pd}/{k}').read_text().strip() for k in ['state','errinfo']} for pd in [0,1]}
summary['source']={name:{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True)} for name,repo in [('library',L),('llama',R/'repos/llama.cpp')]}
summary['frames']={}
D=R/'build/flash-online-lib/lib/kunlunk200/kernels'
for name in ['kunlun-flash-online','kunlun-flash-online-ordered']:
 summary['frames'][name]={'frame':re.findall(r'addi sp, sp, (-\d+)',(D/(name+'.device.s')).read_text()),'bytes':(D/(name+'.bin')).stat().st_size,'sha256':hashlib.sha256((D/(name+'.bin')).read_bytes()).hexdigest()}
(O/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
