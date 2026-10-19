# Biotech event-driven sentiment: does the tone of a press release predict the stock?

An event study and a simple trading backtest on company press releases from 82 listed biotech and pharma names,
2018 to 2026. Every release (FDA decisions, trial readouts, earnings, deals) is scored with FinBERT, and the
question is whether the score says anything about the stock's return relative to the XBI biotech ETF.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_fetch_prices.py` | Daily adjusted closes for the universe and XBI from Financial Modeling Prep (FMP), cached in `data/raw/prices/` |
| 2 | `src/02_fetch_press_releases.py` | Every press release since 2018 per ticker from FMP `/api/v3/press-releases/{ticker}`, cached in `data/raw/press_releases/` |
| 3 | `src/03_sentiment.py` | FinBERT (`ProsusAI/finbert`) on title + first 2,000 characters, score = P(positive) − P(negative); tags each release as fda / trial / earnings / deal / other by title keywords. Output `data/events.csv` |
| 4 | `src/04_backtest.py` | Event study and strategy backtest. Output `output/results.json`, `output/event-study.png`, `output/strategy-equity.png`, `output/events_scored.csv`, `output/trades_long.csv` |

Run in order with `FMP_API_KEY=... python3 src/0N_*.py`. Step 3 takes about an hour on a laptop CPU.

## Method

- **Event day**: the first trading day on or after the release; releases time-stamped after 16:00 roll to the next day.
  Several releases from one company on one day are merged into one event with the mean score.
- **Abnormal return**: stock return minus XBI return (market-adjusted), summed over the window.
  Windows: [-5,+1], [0,+1], [0,+5], [0,+20] trading days. Events are split into sentiment terciles.
- **Strategy**: a release counts as a signal when its score is in the top tercile of the trailing 252 trading days of
  events (so no look-ahead in the cutoff). Buy at the close of day +1 (a realistic earliest fill for an overnight
  model), hold 20 trading days, equal weight across open positions, 10 bp per side. Long-short adds the bottom
  tercile as a short leg. Benchmark: XBI buy and hold over the same dates.

## Caveats

- **Survivorship**: the universe is names listed in 2026, so companies that failed or were acquired are missing.
  That biases returns upward for everyone, benchmark excepted.
- **Text coverage**: FMP's press-release archive starts between 2018 and 2020 depending on the ticker, so the
  early years are thinner.
- **Market-adjusted, not beta-adjusted** abnormal returns; small caps with beta above one will show more noise.
- **The signal is one model's tone score** on the first part of the text. It does not read the numbers.
- **Not FDA approvals from the FDA**: the `fda` tag comes from the company's own announcement of a decision,
  which is what the market trades on, but it also catches filings, designations and complete response letters.
