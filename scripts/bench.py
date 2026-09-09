#!/usr/bin/env python3
"""knock-ocr 并发压测（只用标准库，宿主机不需要装任何东西）

    python3 scripts/bench.py --url http://127.0.0.1:8710/api/ocr \
        --file sample.png --concurrency 8 --requests 40

输出 QPS / P50 / P95 / P99 与错误分布，用来确定后续 worker 数与 batch 规模。
"""
import argparse
import mimetypes
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def build_body(path):
    boundary = "----knockocr" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--file", default=".deploy/sample.png",
                    help="压测用的图片；默认用 manage.sh test 生成的样张")
    ap.add_argument("--concurrency", "-c", type=int, default=8)
    ap.add_argument("--requests", "-n", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()

    path = Path(a.file)
    if not path.exists():
        print(f"找不到 {path}\n先跑一次：bash scripts/manage.sh test", file=sys.stderr)
        return 2

    body, ctype = build_body(path)
    url = a.url + ("&" if "?" in a.url else "?") + "include_json=false"

    lat = []
    errs = {}
    lock = threading.Lock()
    counter = iter(range(a.requests))

    def worker():
        while True:
            try:
                next(counter)
            except StopIteration:
                return
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(url, data=body,
                                             headers={"Content-Type": ctype})
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    r.read()
                ms = (time.perf_counter() - t0) * 1000
                with lock:
                    lat.append(ms)
            except urllib.error.HTTPError as e:
                with lock:
                    errs[f"HTTP {e.code}"] = errs.get(f"HTTP {e.code}", 0) + 1
            except Exception as e:
                key = type(e).__name__
                with lock:
                    errs[key] = errs.get(key, 0) + 1

    print(f"目标 {url}")
    print(f"样本 {path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"并发 {a.concurrency}  总请求 {a.requests}\n")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = len(lat)
    print(f"耗时      {wall:.2f}s")
    print(f"成功/总数 {ok}/{a.requests}")
    if ok:
        s = sorted(lat)
        pct = lambda p: s[min(len(s) - 1, int(len(s) * p))]
        print(f"吞吐      {ok / wall:.2f} req/s")
        print(f"延迟      P50 {pct(.5):.0f}ms  P95 {pct(.95):.0f}ms  "
              f"P99 {pct(.99):.0f}ms  mean {statistics.fmean(s):.0f}ms")
        print(f"          min {s[0]:.0f}ms  max {s[-1]:.0f}ms")
    if errs:
        print("错误      " + "  ".join(f"{k}×{v}" for k, v in sorted(errs.items())))
    print("\n提示：并发从 1 往上扫（1/2/4/8/16），QPS 不再涨的那个点就是当前配置的上限。")
    print("      若 P95 陡增而 QPS 不涨，先加 WORKERS，再考虑加卡。")
    return 0 if ok == a.requests else 1


if __name__ == "__main__":
    sys.exit(main())
