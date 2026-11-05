"""Short-lived canary for the detached supervisor; no model/GPU work."""
import json
import os
from pathlib import Path
import time

out = Path('/mnt/data/thanhpv/ldva/logs/overnight/detach_probe.json')
for index in range(15):
    out.write_text(json.dumps(dict(pid=os.getpid(), ppid=os.getppid(),
        sid=os.getsid(0), pgid=os.getpgrp(), heartbeat=index, timestamp=time.time())))
    time.sleep(2)
