"""Subprocess fixture for runner tests. Never an analysis implementation."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
print("fixture stdout", flush=True)
print("fixture stderr", file=sys.stderr, flush=True)
if mode == "success":
    print("Observed significance (median): 1.9", flush=True)
elif mode == "failure":
    raise SystemExit(7)
elif mode == "sleep":
    time.sleep(60)
elif mode == "descendant":
    child = subprocess.Popen([sys.executable, __file__, "ignore_term", sys.argv[2]])
    child.wait()
elif mode == "orphan":
    subprocess.Popen([sys.executable, __file__, "ignore_term", sys.argv[2]])
elif mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[2]).write_text(str(os.getpid()))
    time.sleep(60)
