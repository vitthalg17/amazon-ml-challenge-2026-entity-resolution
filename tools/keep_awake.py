"""Keep Windows from idle-sleeping while a process runs (no admin rights needed).

    python tools/keep_awake.py <pid>

Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED): the system stays awake (the
display may still turn off) until the watched process exits. Closing the lid can still sleep the
machine, depending on the power settings.
"""
import ctypes
import sys
import time

import psutil

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main():
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        while psutil.pid_exists(pid):
            time.sleep(30)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
