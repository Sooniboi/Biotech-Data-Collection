"""Every company press release since 2018 for the universe, via FMP /api/v3/press-releases/{ticker} (paged, newest first)."""
import json
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, UNIVERSE, START, fmp, cached

d = RAW / "press_releases"; d.mkdir(exist_ok=True)
total = 0
for t in UNIVERSE:
    def fetch():
        out, page = [], 0
        while page < 60:
            batch = fmp(f"/api/v3/press-releases/{t}", page=page)
            if not batch: break
            out += batch
            if batch[-1]["date"] < START: break
            page += 1
        return [r for r in out if r["date"] >= START]
    rel = cached(d / f"{t}.json", fetch) or []
    total += len(rel)
    print(t, len(rel), rel[-1]["date"][:10] if rel else "-", flush=True)
print("total releases", total)
