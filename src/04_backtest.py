"""Event study and a simple sentiment strategy on the scored press releases.

Event study: market-adjusted abnormal returns (stock minus XBI) over [-5,+1], [0,+1], [0,+5], [0,+20] trading
days around each release, split into sentiment terciles. Strategy: buy names whose release scores in the top
tercile of the trailing-year distribution, at the close of day +1, hold 20 trading days, equal weight across open
positions, 10 bp per side; long-short adds a short leg on the bottom tercile. Benchmark: XBI buy and hold.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, ROOT, OUT, BENCH, START, END
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as md

WINDOWS = {"[-5,+1]": (-5, 1), "[0,+1]": (0, 1), "[0,+5]": (0, 5), "[0,+20]": (0, 20)}
HOLD, COST = 20, 0.001
ACCENT, GREY = "#ff5a1f", "#8a8a8a"

# ---- prices -> daily simple returns, wide
px = pd.read_csv(RAW / "prices.csv", parse_dates=["date"]).pivot(index="date", columns="ticker", values="adj_close").sort_index()
px = px.loc[(px.index >= "2017-11-01") & (px.index <= END)]
ret = px.pct_change()
dates = px.index
pos_of = pd.Series(np.arange(len(dates)), index=dates)

# ---- events: one per ticker-day (mean score), mapped to the first trading day on/after the release
ev = pd.read_csv(ROOT / "data" / "events.csv", parse_dates=["date"])
ev = ev[(ev.date >= START) & ev.ticker.isin(px.columns)]
after_close = ev["time"].fillna("00:00") >= "16:00"
ev["effective"] = ev["date"] + pd.to_timedelta(after_close.astype(int), unit="D")
ev = ev.groupby(["ticker", "effective"], as_index=False).agg(score=("score", "mean"), n=("score", "size"), kind=("kind", "first"), title=("title", "first"))
ev["t0"] = dates.searchsorted(ev["effective"].values)          # index of event day
ev = ev[(ev.t0 >= 130) & (ev.t0 < len(dates) - 21)].copy()
ev["t0date"] = dates[ev.t0]

# ---- abnormal returns around each event (market-adjusted)
R = ret.values; b = pos_of
bench = ret[BENCH].values
col = {t: i for i, t in enumerate(px.columns)}
def car(row, a, z):
    i, c = row.t0, col[row.ticker]
    s = R[i + a: i + z + 1, c]; m = bench[i + a: i + z + 1]
    if np.isnan(s).any(): return np.nan
    return float(np.nansum(s - m))
for w, (a, z) in WINDOWS.items():
    ev[w] = [car(r, a, z) for r in ev.itertuples()]
ev = ev.dropna(subset=list(WINDOWS))
ev["tercile"] = pd.qcut(ev.score, 3, labels=["bottom", "middle", "top"])

def summarise(df):
    out = {}
    for w in WINDOWS:
        g = df.groupby("tercile", observed=True)[w]
        out[w] = {k: {"mean": round(100 * v.mean(), 2), "t": round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2) if len(v) > 2 else None, "n": int(len(v))} for k, v in g}
        top, bot = df[df.tercile == "top"][w], df[df.tercile == "bottom"][w]
        diff = top.mean() - bot.mean(); se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
        out[w]["top_minus_bottom"] = {"mean": round(100 * diff, 2), "t": round(float(diff / se), 2)}
    return out
event_study = {"all": summarise(ev)}
for k in ["fda", "trial", "earnings"]:
    sub = ev[ev.kind == k]
    if len(sub) > 60: event_study[k] = summarise(sub)

# ---- strategy: trailing-252-trading-day tercile cutoffs (no look-ahead), enter close of t0+1, hold 20 days
ev = ev.sort_values("t0").reset_index(drop=True)
hist_scores, hist_t = [], []
long_sig, short_sig = [], []
for r in ev.itertuples():
    past = [s for s, t in zip(hist_scores, hist_t) if r.t0 - 252 <= t < r.t0]
    if len(past) >= 100:
        lo, hi = np.percentile(past, [100 / 3, 200 / 3])
        long_sig.append(r.score >= hi); short_sig.append(r.score <= lo)
    else:
        long_sig.append(False); short_sig.append(False)
    hist_scores.append(r.score); hist_t.append(r.t0)
ev["long"], ev["short"] = long_sig, short_sig

def leg(mask):
    """Daily return of an equal-weight book of 20-day positions entered at close of t0+1 (so first return is day t0+2)."""
    n = len(dates); gross = np.zeros(n); cnt = np.zeros(n); trades = []
    for r in ev[mask].itertuples():
        e = r.t0 + 1; c = col[r.ticker]
        rs = R[e + 1: e + 1 + HOLD, c]
        if np.isnan(rs).any(): continue
        gross[e + 1: e + 1 + HOLD] += rs; cnt[e + 1: e + 1 + HOLD] += 1
        pnl = float(np.prod(1 + rs) - 1); bm = float(np.prod(1 + bench[e + 1: e + 1 + HOLD]) - 1)
        trades.append({"ticker": r.ticker, "date": str(r.t0date.date()), "ret": pnl, "excess": pnl - bm})
        gross[e + 1] -= COST; gross[e + HOLD] -= COST   # entry and exit costs charged to the position
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.where(cnt > 0, gross / np.maximum(cnt, 1), 0.0)
    return pd.Series(daily, index=dates), trades

start_i = int(ev[ev.long | ev.short].t0.min()) if (ev.long | ev.short).any() else 260
long_r, long_trades = leg(ev.long)
short_r, short_trades = leg(ev.short)
all_r, all_trades = leg(ev.long | ev.short | ~(ev.long | ev.short))   # control: every event, same mechanics, no sentiment filter
bot_r, bot_trades = leg(ev.short)
ls_r = long_r - short_r
bench_r = ret[BENCH].fillna(0)
sl = slice(start_i, None)
curves = pd.DataFrame({"Long top tercile": long_r, "Long-short": ls_r, "All events (control)": all_r, "Bottom tercile": bot_r, "XBI": bench_r}).iloc[sl]
eq = (1 + curves).cumprod()

def metrics(r, trades=None):
    r = r.dropna(); years = len(r) / 252
    e = (1 + r).cumprod(); cagr = e.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252); sharpe = (r.mean() * 252) / vol if vol > 0 else 0
    dd = (e / e.cummax() - 1).min()
    m = {"cagr": round(100 * cagr, 2), "vol": round(100 * vol, 2), "sharpe": round(float(sharpe), 2), "max_drawdown": round(100 * float(dd), 2), "total_return": round(100 * (e.iloc[-1] - 1), 2)}
    if trades is not None:
        m["trades"] = len(trades); m["hit_rate_vs_xbi"] = round(100 * np.mean([t["excess"] > 0 for t in trades]), 1) if trades else None
        m["avg_excess_per_trade"] = round(100 * np.mean([t["excess"] for t in trades]), 2) if trades else None
    return m

results = {
    "universe": int(px.shape[1] - 1), "events": int(len(ev)), "first_event": str(ev.t0date.min().date()), "last_event": str(ev.t0date.max().date()),
    "events_by_kind": ev.kind.value_counts().to_dict(),
    "event_study_pct": event_study,
    "strategy": {"from": str(dates[start_i].date()), "to": str(dates[-1].date()), "hold_days": HOLD, "cost_bp_per_side": COST * 1e4,
                 "long_top_tercile": metrics(curves["Long top tercile"], long_trades), "long_short": metrics(curves["Long-short"], long_trades + short_trades),
                 "all_events_control": metrics(curves["All events (control)"], all_trades), "long_bottom_tercile": metrics(curves["Bottom tercile"], bot_trades), "xbi": metrics(curves["XBI"])},
}
(OUT / "results.json").write_text(json.dumps(results, indent=1))
ev.drop(columns=["title"]).to_csv(OUT / "events_scored.csv", index=False)
pd.DataFrame(long_trades).to_csv(OUT / "trades_long.csv", index=False)
eq.to_csv(OUT / "equity_curves.csv")

# ---- figures, in the site's style
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": GREY, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY})
def style(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(GREY)
    ax.set_facecolor("none"); ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.5)

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200); fig.patch.set_alpha(0); style(ax)
ws = list(WINDOWS); x = np.arange(len(ws)); wdt = 0.26
shades = {"bottom": "#c8c8c8", "middle": GREY, "top": ACCENT}
for j, terc in enumerate(["bottom", "middle", "top"]):
    means = [event_study["all"][w][terc]["mean"] for w in ws]
    ax.bar(x + (j - 1) * wdt, means, wdt, color=shades[terc], label=f"{terc} sentiment tercile")
ax.axhline(0, color=GREY, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([f"{w} days" for w in ws]); ax.set_ylabel("Mean abnormal return vs XBI, %")
ax.legend(frameon=False, loc="lower right", labelcolor=GREY, fontsize=8, ncol=3)
ax.set_title(f"Press releases scored by FinBERT, {results['events']:,} events, {results['universe']} biotechs, {results['first_event'][:4]}–{results['last_event'][:4]}", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "event-study.png", transparent=True)

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200); fig.patch.set_alpha(0); style(ax)
s = results["strategy"]
ax.plot(eq.index, 100 * eq["XBI"], color=GREY, lw=1.2, label=f"XBI  {s['xbi']['total_return']:+.0f}%")
ax.plot(eq.index, 100 * eq["All events (control)"], color=GREY, lw=1.2, ls="--", label=f"Every release, no sentiment filter  {s['all_events_control']['total_return']:+.0f}%")
ax.plot(eq.index, 100 * eq["Bottom tercile"], color="#c8c8c8", lw=1.2, label=f"Bottom tercile  {s['long_bottom_tercile']['total_return']:+.0f}%")
ax.plot(eq.index, 100 * eq["Long top tercile"], color=ACCENT, lw=1.8, label=f"Long top tercile, 20-day hold  {s['long_top_tercile']['total_return']:+.0f}%")
ax.axhline(100, color=GREY, lw=0.6, ls=":"); ax.xaxis.set_major_locator(md.YearLocator()); ax.xaxis.set_major_formatter(md.DateFormatter("%Y"))
ax.set_ylabel("Value of $100"); ax.legend(frameon=False, loc="upper left", labelcolor=GREY, fontsize=8)
ax.set_title(f"Backtest {s['from']} to {s['to']}, equal weight, 10 bp per side", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "strategy-equity.png", transparent=True)
print(json.dumps(results, indent=1)[:3000])
