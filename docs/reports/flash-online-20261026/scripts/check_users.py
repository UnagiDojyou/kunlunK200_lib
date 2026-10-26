"""Read all process FDs with sudo; fail closed if any cannot be inspected."""
from pathlib import Path
import os,sys
users=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        for fd in (proc/'fd').iterdir():
            try:target=os.readlink(fd)
            except FileNotFoundError:continue
            if target.startswith('/dev/xpu'):users.append((proc.name,fd.name,target,(proc/'comm').read_text().strip()))
    except FileNotFoundError:continue
    except PermissionError:print('Cannot inspect '+str(proc),file=sys.stderr);sys.exit(2)
for row in users:print(*row)
sys.exit(1 if users else 0)
