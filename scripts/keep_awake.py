"""Keep the Windows host awake while long experiments run.

The training grid was once suspended by system sleep for several hours, which
produced a meaningless 22,540 s epoch. This helper holds ES_SYSTEM_REQUIRED for
the lifetime of the process.
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

kernel32 = ctypes.windll.kernel32
kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("keep-awake active", flush=True)
try:
    while True:
        time.sleep(60)
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except KeyboardInterrupt:
    kernel32.SetThreadExecutionState(ES_CONTINUOUS)
