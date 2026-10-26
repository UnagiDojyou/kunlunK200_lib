from pathlib import Path
import subprocess,json,shutil,tarfile,hashlib,io,tempfile,os
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200';P=R/'repos/llama.cpp';D=O/'delivery'
assert (O/'FINAL_TESTS_DONE').exists() and not (O/'XPU_ABORT').exists()
v=json.loads((O/'final-verification.json').read_text());assert all(all(d.values()) for d in v['baseline_production_kernels_unchanged'].values())
for pd in [0,1]:
 assert Path(f'/proc/xpu/dev{pd}/state').read_text().strip()=='RUNNING'
 assert Path(f'/proc/xpu/dev{pd}/errinfo').read_text().strip()=='no error'
with tarfile.open(O/'client-scripts.tar.gz') as t:t.extractall(D,filter='data')
shutil.copy2(O/'REPRODUCE.ja.md',D/'REPRODUCE.ja.md')
for name in ['runner.py','check_users.py','quality.py','audit.cpp']:
 src=(O/name) if (O/name).exists() else R/'results/flash-attention-20261026'/name
 shutil.copy2(src,D/'client-scripts'/name)
data=D/'data';data.mkdir(exist_ok=True)
for pat in ['*.json','*.csv']:
 for f in O.glob(pat):shutil.copy2(f,data/f.name)
shutil.copytree(O/'inputs',D/'inputs',dirs_exist_ok=True)
shutil.copytree(O/'logs',D/'logs',dirs_exist_ok=True)
# Keep error evidence without duplicating unrelated historical build trees.
incident=D/'incident';incident.mkdir(exist_ok=True)
for d in (O/'incidents').iterdir():
 dst=incident/d.name;dst.mkdir(exist_ok=True)
 for f in d.iterdir():
  if f.is_file() and f.suffix in ['.txt','.json','.patch','.binary']:shutil.copy2(f,dst/f.name)
 for name in ['kernel-sources','recovery']:
  if (d/name).is_dir():shutil.copytree(d/name,dst/name,dirs_exist_ok=True)
# Store all logits hashes; retain two small selected raw samples, not >GB of corpus dumps.
manifest={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in O.glob('*.bin')}
(data/'raw-logits-manifest.json').write_text(json.dumps(manifest,indent=2))
samples=D/'samples';samples.mkdir(exist_ok=True)
for mode in ['off','auto']:shutil.copy2(O/f'final-auto-tiny-p0.bin.p4096.{mode}',samples/f'tiny-tg4096-{mode}.bin')
for f in O.glob('*.bin.p*'):
 manifest[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(data/'raw-logits-manifest.json').write_text(json.dumps(manifest,indent=2))
# Keep the report in both repositories, with compact evidence and reproduction scripts.
doc=L/'docs/reports/flash-decoding-20261026';doc.mkdir(parents=True,exist_ok=True)
for name in ['REPORT.ja.md','REPRODUCE.ja.md']:shutil.copy2(D/name,doc/name)
shutil.copytree(data,doc/'data',dirs_exist_ok=True);shutil.copytree(D/'client-scripts',doc/'scripts',dirs_exist_ok=True)
q=L/'docs/flash-decoding.md';q.write_text(q.read_text().replace('F16 broadcast causal-style mask','F16 mask'))
q=P/'docs/backend/kunlun-flash-decoding.md';q.write_text(q.read_text().replace('KunlunK2000.7.3','KunlunK200 0.7.3'))
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True)
# Only task-owned paths are staged. Keep main/master and remotes untouched.
lib_paths=['CMakeLists.txt','docs/flash-attention.md','src/flash.inc','src/flash-decode.inc','src/flash-profile.inc','bench/flash-decode-bench.cpp','docs/flash-decoding.md','docs/reports/flash-decoding-20261026','kernels/flash-decode-config.h']
lib_paths += [str(p.relative_to(L)) for pattern in ['kunlun-flash-decode*.xpu','kunlun-flash-profile*.xpu'] for p in (L/'kernels').glob(pattern)]
assert git(L,'branch','--show-current').strip()=='codex/k200-flash-decoding-20261026'
subprocess.run(['git','add','--',*lib_paths],cwd=L,check=True)
subprocess.run(['git','commit','-m','kunlun: add validated Flash-Decoding 0.7.3 with conservative TG dispatch\n\nAssisted-by: OpenAI Codex'],cwd=L,check=True)
parent_paths=['examples/kunlun/CMakeLists.txt','examples/kunlun/flash-decoding-bench.cpp','docs/backend/kunlun-flash-decoding.md','vendor/KunlunK200']
assert git(P,'branch','--show-current').strip()=='codex/k200-flash-decoding-20261026'
subprocess.run(['git','add','--',*parent_paths],cwd=P,check=True)
subprocess.run(['git','commit','-m','kunlun: integrate Flash-Decoding 0.7.3 and controlled TG benchmark\n\nAssisted-by: OpenAI Codex'],cwd=P,check=True)
assert not git(L,'status','--porcelain').strip() and not git(P,'status','--porcelain').strip()
commits={n:dict(head=git(p,'rev-parse','HEAD').strip(),branch=git(p,'branch','--show-current').strip(),status=git(p,'status','--short'),pushed=False) for n,p in [('library',L),('llama',P)]}
(O/'final-commits.json').write_text(json.dumps(commits,indent=2));(data/'final-commits.json').write_text(json.dumps(commits,indent=2))
patches=D/'patches';patches.mkdir(exist_ok=True);sources=D/'source';sources.mkdir(exist_ok=True);baselines=D/'baseline-source';baselines.mkdir(exist_ok=True)
bases={'library':'d8bd87f56ae95872287c4f4a030b7ac7e68dd7b9','llama':'8d62617cb1d267d9283d5fcffe8e66c57820dfcc'}
proof={}
for name,repo in [('library',L),('llama',P)]:
 base=bases[name]
 patch=subprocess.check_output(['git','diff','--binary',base,'HEAD'],cwd=repo);(patches/(name+'.patch')).write_bytes(patch)
 if name=='llama':patch=subprocess.check_output(['git','diff','--binary',base,'HEAD','--','.',':(exclude)vendor/KunlunK200'],cwd=repo);(patches/'llama-source.patch').write_bytes(patch)
 archive=subprocess.check_output(['git','archive','--format=tar',base],cwd=repo)
 with tarfile.open(baselines/(name+'.tar.gz'),'w:gz') as t:
  with tarfile.open(fileobj=io.BytesIO(archive)) as src:
   for member in src.getmembers():t.addfile(member,src.extractfile(member) if member.isfile() else None)
 with tempfile.TemporaryDirectory(prefix='verify-fd-',dir=O) as tmp:
  tmp=Path(tmp)
  with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(tmp,filter='data')
  patchfile=patches/(name+'.patch' if name=='library' else 'llama-source.patch')
  subprocess.run(['git','apply','--check',str(patchfile)],cwd=tmp,check=True);subprocess.run(['git','apply',str(patchfile)],cwd=tmp,check=True)
  names=git(repo,'ls-files').splitlines();count=0
  for rel in names:
   f=repo/rel;other=tmp/rel
   if f.is_symlink():assert other.is_symlink() and os.readlink(f)==os.readlink(other)
   elif f.is_file():assert other.is_file() and hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(other.read_bytes()).digest()
   else:continue
   count+=1
  proof[name]=dict(files=count,baseline=base,commit=commits[name]['head'],patch_apply_and_hash_equal=True)
 with tarfile.open(sources/(name+'.tar.gz'),'w:gz') as t:
  for rel in git(repo,'ls-files').splitlines():
   f=repo/rel
   if f.is_file() or f.is_symlink():t.add(f,arcname=rel,recursive=False)
 # Directly viewable changed source files, excluding evidence already at delivery root.
 root=sources/name
 for rel in git(repo,'diff','--name-only',base,'HEAD').splitlines():
  if rel.startswith('docs/reports/'):continue
  f=repo/rel
  if f.is_file() and not f.is_symlink():dst=root/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,dst)
bins=D/'binaries';bins.mkdir(exist_ok=True)
for label,build,sub in [('baseline','flash-kq-lib',''),('new','flash-decode-lib',''),('new-llama','flash-decode-llama','kunlunk200')]:
 path=R/'build'/build/sub/'lib';dst=bins/label;dst.mkdir(exist_ok=True)
 shutil.copy2(path/'libkunlunk200.so',dst/'libkunlunk200.so')
 kd=dst/'kernels';kd.mkdir(exist_ok=True)
 for pat in ['*.bin','*.entry']:
  for f in (path/'kunlunk200/kernels').glob(pat):shutil.copy2(f,kd/f.name)
(data/'patch-verification.json').write_text(json.dumps(proof,indent=2))
files={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in D.rglob('*') if p.is_file()}
(D/'SHA256.json').write_text(json.dumps(files,indent=2))
with tarfile.open(O/'flash-decoding-delivery-20261026.tar.gz','w:gz') as t:t.add(D,arcname='flash-decoding-delivery-20261026')
print('FINISHED',json.dumps(commits),json.dumps(proof),(O/'flash-decoding-delivery-20261026.tar.gz').stat().st_size,flush=True)
