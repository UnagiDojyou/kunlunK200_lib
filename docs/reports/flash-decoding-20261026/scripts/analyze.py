from pathlib import Path
import json,re,tarfile
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
rows=[]
for p in sorted((O/'logs').glob('*.txt')):
 s=p.read_text(errors='replace')
 if 'FLASH_UNIT' not in s:continue
 f=fields(next(l for l in s.splitlines() if l.startswith('FLASH_UNIT')));f['name']=p.stem
 for k,v in list(f.items()):
  try:f[k]=float(v) if any(c in str(v) for c in '.e') else int(v)
  except ValueError:pass
 fd=next((l for l in s.splitlines() if l.startswith('KUNLUN Flash-Decoding')),None)
 if fd:f['decode']=fields(fd)
 rows.append(f)
(O/'unit-summary.json').write_text(json.dumps(rows,indent=2));print('UNIT',len(rows))
for hk,d in [(1,128),(4,64),(4,128),(32,128)]:
 for seq in [512,2048,4096,16384]:
  rr=[r for r in rows if r['name'].startswith(('grid-p0','long-p0')) and r['D']==d and r['HK']==hk and r['S']==seq]
  if rr:
   r=min(rr,key=lambda x:x['flash_ms']);print('BEST',d,hk,seq,r['name'],r['old_ms'],r['flash_ms'],(r['old_ms']/r['flash_ms']-1)*100)
profiles={}
for pd in [0,1]:
 for hk in [1,4,32]:
  for c in [1,4]:
   name=f'profile-p{pd}-hk{hk}-c{c}';s=(O/'logs'/(name+'.txt')).read_text();modes=[];current={'stages':[],'workers':[]}
   for line in s.splitlines():
    if line.startswith(('DECODE_STAGE','FLASH_STAGE')):current['stages'].append(fields(line))
    elif line.startswith('FLASH_WORK'):current['workers'].append(fields(line))
    elif line.startswith('KUNLUN Flash'):
     current['banner']=line;modes.append(current);current={'stages':[],'workers':[]}
   profiles[name]=modes
(O/'profile-summary.json').write_text(json.dumps(profiles,indent=2))
for n in ['profile-p0-hk1-c4','profile-p0-hk4-c4','profile-p0-hk32-c4']:
 for r in profiles[n][:2]:
  print(n,r.get('banner'))
  for stage in ['kq','online','pv','reduce','final']:
   a=[w for w in r['workers'] if w['stage']==stage]
   if not a:continue
   if a[0]['kind']=='sd':print(stage,'nsamples',len(a),'cmdQ',sum(int(w['t10']) for w in a),'cmdK',sum(int(w['t11']) for w in a),'mac',sum(int(w['t14']) for w in a),'ticks',sum(int(w['t9']) for w in a))
   else:print(stage,'active',sum(int(w['rows'])>0 for w in a),'rows',sum(int(w['rows']) for w in a),'max-ticks',max(int(w['ticks']) for w in a))
with tarfile.open(O/'analysis.tar.gz','w:gz') as t:
 for f in ['unit-summary.json','profile-summary.json','grid-results.json','baseline-commits.json','first-llm-quality.json']:t.add(O/f,arcname=f)
 for f in (L/'kernels').glob('*decode*'):t.add(f,arcname='source/'+f.name)
