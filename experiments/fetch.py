"""Resumable downloader (stdlib only).

The connection to download.pytorch.org resets every 30-90 s here.  curl's
--retry restarts the whole transfer instead of resuming, so it never got past
~85 MB.  This reopens the request with a Range header from wherever the local
file ended, so every attempt keeps the bytes already on disk.
"""

import os
import sys
import time
import urllib.request
import urllib.error

CHUNK = 1 << 20


def fetch(url, path, max_attempts=1000):
    for attempt in range(max_attempts):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        req = urllib.request.Request(url, headers={
            "User-Agent": "python-urllib",
            "Accept-Encoding": "identity",
        })
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    print(f"  server ignored Range (status {r.status}); restarting")
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = int(r.headers.get("Content-Length", 0)) + have
                t0, last = time.time(), have
                with open(path, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        if have - last > 50 * CHUNK:
                            dt = time.time() - t0
                            sp = (have - (last if dt else have)) / max(dt, 1e-9) / 1e6
                            print(f"  {have/1e6:8.1f} / {total/1e6:.1f} MB "
                                  f"({100*have/max(total,1):5.1f} %)  {sp:.1f} MB/s",
                                  flush=True)
                            last = have
            size = os.path.getsize(path)
            if total and size >= total:
                print(f"  done: {size/1e6:.1f} MB after {attempt+1} attempt(s)")
                return size
            print(f"  stream ended early at {size/1e6:.1f} MB; resuming")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  attempt {attempt+1}: {type(e).__name__}: {e}; "
                  f"have {os.path.getsize(path)/1e6 if os.path.exists(path) else 0:.1f} MB",
                  flush=True)
        time.sleep(2)
    raise RuntimeError("download did not complete")


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2])
