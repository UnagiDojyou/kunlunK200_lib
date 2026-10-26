"""Bounded tests with existing thermal guard and device-state fail-stop."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,threading,time
ROOT=Path(os.environ.get('K200_WORKSPACE','/home/unagidojyou/kunlun-k200'))
OUT=Path(__file__).resolve().parents[1]
def states():return {pd:Path(f'/proc/xpu/dev{pd}/state').read_text().strip() for pd in [0,1]}
def run(name,command,env=None,seconds=180):
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    if (OUT/'XPU_ABORT').exists() or (ROOT/'results/THERMAL_ABORT').exists():raise RuntimeError('Persistent abort latch')
    if any(v!='RUNNING' for v in states().values()):raise RuntimeError('Device not RUNNING')
    subprocess.run(['sudo','-n','python3',str(Path(__file__).parent/'check_users.py')],check=True)
    prior=list(logs.glob(name+'*'))
    if prior:
        import shutil
        history=OUT/'log-history'/(name+'-'+str(time.time_ns()));history.mkdir(parents=True)
        for p in prior:shutil.move(p,history/p.name)
    args=list(map(str,command));done=threading.Event()
    def watch():
        while not done.wait(.1):
            if all(v=='RUNNING' for v in states().values()):continue
            (OUT/'XPU_ABORT').write_text('device state error during '+name+'\n')
            # Kill only this test's exact argv, after verifying ancestry.
            for proc in Path('/proc').iterdir():
                if not proc.name.isdecimal():continue
                try:
                    argv=[x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x]
                    if argv!=args:continue
                    pid=int(proc.name);ancestor=pid
                    for _ in range(8):
                        stat=Path(f'/proc/{ancestor}/stat').read_text();ancestor=int(stat[stat.rfind(')')+2:].split()[1])
                        if ancestor==os.getpid():os.kill(pid,signal.SIGKILL);break
                        if ancestor<=1:break
                except (OSError,ValueError):continue
            return
    e=os.environ.copy();e.update(env or {});e['KUNLUN_LOG_DIR']=str(logs)
    monitor=threading.Thread(target=watch,daemon=True);monitor.start();begin=time.monotonic()
    print('RUN',name,flush=True)
    try:
        with (logs/(name+'.txt')).open('w') as f:
            rc=subprocess.run([str(ROOT/'install/bin/k200-thermal-guard'),'--name',name,'--seconds',str(seconds),'--',*args],env=e,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
    finally:done.set();monitor.join(timeout=1)
    final=states();failed=rc!=0 or any(v!='RUNNING' for v in final.values())
    meta={'name':name,'command':args,'environment':env or {},'returncode':rc,'elapsed_s':time.monotonic()-begin,'states':final,
          'binary_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args[:2] if Path(p).is_file()}}
    (logs/(name+'.json')).write_text(json.dumps(meta,indent=2))
    text=(logs/(name+'.txt')).read_text(errors='replace')
    if failed:
        (OUT/'XPU_ABORT').write_text(json.dumps(meta,indent=2))
        with (logs/(name+'-dmesg.txt')).open('w') as f:subprocess.run(['sudo','-n','dmesg','-T'],stdout=f,stderr=subprocess.STDOUT)
        import shutil
        incident=OUT/'incidents'/name;incident.mkdir(parents=True,exist_ok=True)
        for pd in [0,1]:
            for item in ['state','errinfo','errtask','sessions']:
                p=Path(f'/proc/xpu/dev{pd}')/item
                if p.exists():(incident/f'pd{pd}-{item}.txt').write_text(p.read_text())
        for root in [ROOT/'repos/KunlunK200',ROOT/'repos/llama.cpp']:
            with (incident/(root.name+'-diff.patch')).open('w') as f:subprocess.run(['git','diff','HEAD','--binary'],cwd=root,stdout=f)
        for i,p in enumerate(args[:2]):
            if Path(p).is_file():shutil.copy2(p,incident/f'command-{i}.binary')
        shutil.copytree(ROOT/'repos/KunlunK200/kernels',incident/'kernel-sources',dirs_exist_ok=True)
        for build in ['common-integration-lib','common-integration-llama/kunlunk200','set-rows-lib','set-rows-llama/kunlunk200','glu-lib','glu-llama/kunlunk200','attention-lib','attention-llama/kunlunk200']:
            directory=ROOT/'build'/build/'lib'
            tag=build.replace('/','-')
            if (directory/'kunlunk200/kernels').exists():shutil.copytree(directory/'kunlunk200/kernels',incident/('kernel-build-'+tag),dirs_exist_ok=True)
            library=directory/'libkunlunk200.so'
            if library.exists():shutil.copy2(library,incident/('libkunlunk200-'+tag+'.so'))
        print(text[-3000:],flush=True);raise RuntimeError(f'{name} failed; later tests stopped')
    return text
