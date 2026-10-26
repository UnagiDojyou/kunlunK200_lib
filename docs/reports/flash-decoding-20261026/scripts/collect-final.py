from pathlib import Path
import json,re,statistics,csv,runpy,hashlib,subprocess,array,math
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
assert (O/'FINAL_TESTS_DONE').exists() and not (O/'XPU_ABORT').exists()
runpy.run_path(str(O/'collect.py'),run_name='__main__')
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
   w=csv.DictWriter(f,list(dict.fromkeys(k for row in data for k in row)));w.writeheader();w.writerows(data)
ab=[];trials=[];split=[];memory=[];profiles=[]
for p in sorted((O/'logs').glob('*.json')):
 meta=json.loads(p.read_text());name=p.stem;text=p.with_suffix('.txt').read_text(errors='replace');env=meta['environment']
 if name.startswith(('ab-','work-ab-','fixed-ab-','final-auto-')):
  for flag in ['LD_PRELOAD','KUNLUN_FLASH_TRACE','KUNLUN_FLASH_DECODING_PROFILE','K200_CHECK_STATE','K200_TRACE_OPS']:assert flag not in env
  groups={}
  for line in text.splitlines():
   if line.startswith('FDBENCH '):
    x=dict(test=name,model='tiny' if '-tiny-' in name else '7b',pd=int(meta['command'][-1]),**fields(line));trials.append(x);groups.setdefault(x['mode'],[]).append(x['tps'])
  for mode,v in groups.items():ab.append(dict(test=name,model='tiny' if '-tiny-' in name else '7b',pd=int(meta['command'][-1]),mode=mode,median=statistics.median(v),minimum=min(v),maximum=max(v),trials=len(v),gain_pct=100*(statistics.median(v)/statistics.median(groups['off'])-1)))
 if name.startswith('final-split-'):
  for flag in ['LD_PRELOAD','KUNLUN_FLASH_TRACE','KUNLUN_FLASH_DECODING_PROFILE','K200_CHECK_STATE','K200_TRACE_OPS']:assert flag not in env
  for line in text.splitlines():
   if line.startswith('K200PERF '):
    d=fields(line);split.append(dict(test=name,model='tiny' if '-tiny-' in name else '7b',split=meta['command'][2],mode=env['KUNLUN_FLASH_DECODING'],context=d['pp_tokens'],pp_tps=d['pp_tokens']*1000/d['pp_ms'],tg_tps=d['tg_steps']*1000/d['tg_ms']))
 if name.startswith('final-memory-'):
  peak={};transfers=[]
  for line in text.splitlines():
   if line.startswith('FLASH_MEMORY '):
    d=fields(line);pd=d['pd'];peak.setdefault(pd,{'hbm_peak':0,'l3_peak':0});peak[pd]['hbm_peak']=max(peak[pd]['hbm_peak'],d['hbm_peak']);peak[pd]['l3_peak']=max(peak[pd]['l3_peak'],d['l3_peak'])
   if line.startswith('FLASH_AUDIT '):transfers.append(fields(line))
  memory.append(dict(test=name,peaks=peak,counters=transfers))
 if name.startswith('profile-'):
  current={'stage_ms':{},'workers':[]};modes={}
  for line in text.splitlines():
   if line.startswith(('FLASH_STAGE ','DECODE_STAGE ')):
    d=fields(line);current['stage_ms'][d['stage']]=current['stage_ms'].get(d['stage'],0)+d['ms']
   elif line.startswith('FLASH_WORK '):current['workers'].append(fields(line))
   elif line.startswith('KUNLUN Flash'):
    mode='decode' if 'Flash-Decoding' in line else 'legacy';current['banner']=line;modes[mode]=current;current={'stage_ms':{},'workers':[]}
  for mode,r in modes.items():
   row=dict(test=name,mode=mode,banner=r['banner'],stages={})
   for stage in ['kq','online','pv','reduce','final']:
    a=[w for w in r['workers'] if w['stage']==stage]
    if not a:continue
    if a[0]['kind']=='sd':
     groups=[a[i:i+4] for i in range(0,len(a),4)];d={f't{i}':sum(w[f't{i}'] for w in a) for i in range(16)};d.update(launches=len(groups),active=[sum(w['t14']>0 for w in g) for g in groups],critical_ms=sum(max(w['t9'] for w in g) for g in groups)/900000,phase_ms=[sum(max(w[f't{i}'] for w in g) for g in groups)/900000 for i in range(9)])
    else:
     groups=[a[i:i+64] for i in range(0,len(a),64)];d=dict(launches=len(groups),active=[sum(w['rows']>0 for w in g) for g in groups],rows=sum(w['rows'] for w in a),elements=sum(w['elems'] for w in a),critical_ms=sum(max(w['ticks'] for w in g) for g in groups)/900000)
    row['stages'][stage]=d
   profiles.append(row)
for n,d in [('ab-trials',trials),('ab-performance',ab),('split-performance',split),('final-memory',memory),('profiles-final',profiles)]:save(n,d)
teacher=list(map(int,(O/'inputs/teacher64.ids').read_text().split()));chosen=[]
for pd in [0,1]:
 stem=f'final-auto-tiny-p{pd}.bin.p4096';aa=array.array('f');aa.frombytes((O/(stem+'.off')).read_bytes());bb=array.array('f');bb.frombytes((O/(stem+'.auto')).read_bytes());nll=[0.,0.];ma=0.;sq=en=0.;kl=[];greedy=0;changed=[]
 for i in range(64):
  a=aa[i*32000:(i+1)*32000];b=bb[i*32000:(i+1)*32000];xa=max(a);xb=max(b);la=xa+math.log(sum(math.exp(v-xa) for v in a));lb=xb+math.log(sum(math.exp(v-xb) for v in b));nll[0]+=la-a[teacher[i]];nll[1]+=lb-b[teacher[i]];kl.append(sum(math.exp(v-la)*(v-la-w+lb) for v,w in zip(a,b)));same=a.index(xa)==b.index(xb);greedy+=same
  if not same:changed.append(i)
  for v,w in zip(a,b):ma=max(ma,abs(v-w));sq+=(v-w)**2;en+=v*v
 chosen.append(dict(pd=pd,nmse=sq/en,max_abs=ma,mean_KL=sum(kl)/64,max_KL=max(kl),greedy_equal=greedy,positions=64,old_continuation_ppl=math.exp(nll[0]/64),new_continuation_ppl=math.exp(nll[1]/64),delta_nll=(nll[1]-nll[0])/64,changed=changed))
save('selected-quality',chosen)
baseline=json.loads((O/'flash-kq-lib-images.json').read_text());expected={n:h for n,h in baseline.items() if (L/'kernels'/(n[:-4]+'.xpu')).exists()};assert len(expected)==52
images={};frames={}
for build,sub in [('flash-decode-lib',''),('flash-decode-llama','kunlunk200')]:
 D=R/'build'/build/sub/'lib/kunlunk200/kernels';images[build]={n:hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in expected.items()};assert all(images[build].values())
 if not sub:
  for p in D.glob('*decode*.device.s'):
   a=re.findall(r'addi sp, sp, (-\d+)',p.read_text());frames[p.stem]=dict(frames=a,bytes=(D/(p.name.replace('.device.s','.bin'))).stat().st_size)
states={pd:{k:Path(f'/proc/xpu/dev{pd}/{k}').read_text().strip() for k in ['state','errinfo']} for pd in [0,1]};assert all(d['state']=='RUNNING' and d['errinfo']=='no error' for d in states.values())
verification=dict(states=states,resets=1,baseline_production_kernels_unchanged=images,frames=frames,source={n:dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=p,text=True)) for n,p in [('library',L),('llama',R/'repos/llama.cpp')]})
save('final-verification',verification)
print('FINAL COLLECT COMPLETE',len(ab),len(profiles),len(chosen))
