#!/usr/bin/env python3
"""Run one bounded test with temperature checks and persistent abort latch."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--seconds', type=float, default=60)
p.add_argument('command', nargs=argparse.REMAINDER)
a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
root=Path(os.environ.get('KUNLUN_LOG_DIR',str(Path.cwd()/'results'))).resolve()
root.mkdir(parents=True,exist_ok=True)
latch=root/'THERMAL_ABORT'
if latch.exists():
    print(latch.read_text(), flush=True); sys.exit(90)
def sample():
    r=subprocess.run([str(Path(os.environ.get('XRE','/usr/local/xpu-4.33.0'))/'bin/xpu_smi'),'-m'],capture_output=True,text=True,timeout=3)
    rows=[]
    for line in r.stdout.splitlines():
        v=line.split()
        if len(v)>=22:
            try: rows.append({'dev':int(v[2]),'die':float(v[4]),'hbm':float(v[6]),'power_mw':float(v[8]),'mhz':float(v[9]),'raw':line})
            except ValueError: pass
    if r.returncode or len(rows)!=2: raise RuntimeError('temperature unavailable: '+r.stdout+r.stderr)
    for row in rows:
        if not all(0 < row[k] < 150 for k in ('die','hbm')): raise RuntimeError('invalid temperature '+str(row))
    return rows

proc=None
peak=0
t0=time.monotonic()
log=open(root/f'{a.name}-thermal.jsonl','a',buffering=1)
def stop():
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait(timeout=2)
try:
    while True:
        rows=sample()
        elapsed=time.monotonic()-t0
        log.write(json.dumps({'elapsed':elapsed,'rows':rows})+'\n')
        temp=max(row[k] for row in rows for k in ('die','hbm'))
        peak=max(peak,temp)
        if temp>=80:
            msg=f'THERMAL ABORT: {temp} C >= 80 C, test={a.name}'
            latch.write_text(msg+'\n')
            print(msg,flush=True); stop(); sys.exit(90)
        if proc is None:
            print(f'GUARD START {a.name} temp={temp} C command={cmd}',flush=True)
            proc=subprocess.Popen(cmd,start_new_session=True)
        elif proc.poll() is not None:
            print(f'GUARD END {a.name} rc={proc.returncode} peak={peak} C elapsed={elapsed:.2f}s',flush=True)
            sys.exit(proc.returncode)
        if elapsed>a.seconds:
            print(f'TIMEOUT {a.name} elapsed={elapsed:.2f}s peak={peak} C',flush=True); stop(); sys.exit(124)
        time.sleep(.5)
except (Exception,KeyboardInterrupt) as exc:
    print('GUARD ERROR: '+str(exc),flush=True); stop(); sys.exit(91)
finally:
    log.close()
