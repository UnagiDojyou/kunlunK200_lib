from pathlib import Path
import subprocess,json,tarfile,shutil,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';O.mkdir(exist_ok=True)
for name,repo in [('library',R/'repos/KunlunK200'),('llama',R/'repos/llama.cpp')]:
 def git(*a):return subprocess.check_output(['git',*a],cwd=repo,text=True)
 record={k:git(*a) for k,a in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'remotes':['remote','-v']}.items()}
 print(name,json.dumps(record));print(git('diff','--stat'));print(git('ls-files','--others','--exclude-standard'))
 (O/(name+'-before.json')).write_text(json.dumps(record,indent=2));(O/(name+'-before.patch')).write_bytes(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=repo))
 with tarfile.open(O/(name+'-before.tar.gz'),'w:gz') as t:
  for rel in git('ls-files','--cached','--others','--exclude-standard').splitlines():
   p=repo/rel
   if p.is_file() and not p.is_symlink():t.add(p,arcname=rel)
 for p in repo.rglob('AGENTS.md'):
  if 'vendor' not in p.parts:print('INSTRUCTIONS',p,p.read_text())
A=R/'results/flash-kq-20261026'
for f in ['runner.py','quality.py','check_users.py']:shutil.copy2(A/f,O/f)
shutil.copytree(A/'inputs',O/'inputs',dirs_exist_ok=True)
subprocess.run(['sudo','-n','python3',str(O/'check_users.py')],check=True)
print(subprocess.check_output([str(R/'external/xre/bin/xpu_smi')],text=True))
for build,sub in [('flash-kq-lib',''),('flash-kq-llama','kunlunk200')]:
 D=R/'build'/build/sub/'lib/kunlunk200/kernels';(O/(build+'-images.json')).write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.bin')},indent=2))
