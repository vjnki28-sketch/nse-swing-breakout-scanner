# NSE Swing Breakout Scanner v2.0

A GitHub + Streamlit end-of-day scanner for NSE equity stocks, designed for swing-trading research using daily candles.

## What's new in v2
- Swing-only daily analysis with up to two years of OHLCV history and 55-session breakout levels.
- Batch downloads to reduce the risk of oversized Yahoo Finance requests.
- Ten explicit factor scores that add to 100 points.
- More conservative signal labels: Confirmed breakout, Successful retest candidate, Near breakout, Developing setup, and Not ready.
- Daily chart candidates: ascending/symmetrical/descending triangles, bull flags, tightening bases and double bottoms.
- NIFTY 50 and a limited, explicit mapping of common symbols to NSE sector indices. Unknown sectors are labelled unmapped and scored neutrally.
- Optional recent-news headline count for the top 40 technical candidates.
- Candlestick chart with support/resistance, factor audit, shortlist and CSV exports.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository, e.g. `nse-swing-breakout-scanner`.
2. Upload `app.py`, `scanner.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Create an app, select your repository and branch, and set `app.py` as the main file.
5. Deploy.

## Daily workflow
Run the app after the NSE cash market closes (after 3:30 PM IST), once the data provider has published the completed daily candle. Click **Run end-of-day scan**. This is a manual end-of-day scan, not an automatically scheduled job. The displayed latest-bar date helps you check the data date.

## Data and limitations
- Price history comes from Yahoo Finance (`SYMBOL.NS`) and is not an exchange-certified feed. Coverage, latency, rate limits, corporate actions and missing symbols can vary.
- NSE's equity list is used for the universe; scanning every symbol can take time. Start with 300–500 symbols if your provider/network is rate-limited.
- Sector-index mapping is intentionally limited to selected common symbols. Unmapped stocks receive a neutral sector score; the app does not pretend to know their sector.
- News is optional and only counts Google News RSS headlines. It does not classify sentiment, verify materiality or establish a causal catalyst.
- Pattern detection and all factor scores are heuristic. A score is not a probability of success, and a breakout label is not a trade instruction.
- Backtest and paper-trade the rules. Account for liquidity, slippage, brokerage, taxes, corporate actions and survivorship bias before any live use.

## Push to GitHub

```bash
git init
git add .
git commit -m "Upgrade NSE swing breakout scanner v2"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/nse-swing-breakout-scanner.git
git push -u origin main
```

Not investment advice. Use at your own risk.
