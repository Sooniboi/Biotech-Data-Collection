"""Daily adjusted closes for the universe and the XBI benchmark from FMP, cached in data/raw/prices."""
import json, sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, UNIVERSE, BENCH, START, END, fmp, cached

d = RAW / "prices"; d.mkdir(exist_ok=True)
rows = []
for t in [BENCH] + UNIVERSE:
    data = cached(d / f"{t}.json", lambda: fmp(f"/api/v3/historical-price-full/{t}", **{"from": "2017-11-01", "to": END}))
    hist = (data or {}).get("historical") or []
    for h in hist:
        rows.append((t, h["date"], h.get("adjClose", h["close"])))
    print(t, len(hist), hist[-1]["date"] if hist else "-", hist[0]["date"] if hist else "-", flush=True)
import csv
with open(RAW / "prices.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ticker", "date", "adj_close"]); w.writerows(rows)
print("rows", len(rows))
