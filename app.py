from __future__ import annotations
import io
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from scanner import WEIGHTS, SECTOR_MAP, analyse_stock, index_strength

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NSE Swing Breakout Scanner", page_icon="📈", layout="wide")
NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_universe():
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    response = requests.get(NSE_LIST_URL, headers=headers, timeout=25)
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns:
        raise ValueError("NSE symbol file did not contain a SYMBOL column.")
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
    symbols = sorted(set(df["SYMBOL"].dropna().astype(str).str.strip()))
    return [s for s in symbols if s and s.upper() not in {"NAN", "NULL"}]

@st.cache_data(ttl=8 * 3600, show_spinner=False)
def download_daily_history(symbols, period="2y"):
    """Download daily adjusted OHLCV in small batches to reduce provider failures."""
    result = {}
    symbols = list(symbols)
    for start in range(0, len(symbols), 60):
        batch = symbols[start:start + 60]
        tickers = [f"{s}.NS" for s in batch]
        raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=False, timeout=30)
        for symbol, ticker in zip(batch, tickers):
            try:
                frame = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                frame.columns = [str(c).title() for c in frame.columns]
                needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in frame.columns for c in needed):
                    frame = frame[needed].dropna(subset=["Close"])
                    if len(frame) >= 210:
                        result[symbol] = frame
            except Exception:
                continue
        time.sleep(0.15)
    return result

@st.cache_data(ttl=8 * 3600, show_spinner=False)
def download_index_data(ticker, period="2y"):
    try:
        frame = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(c).title() for c in frame.columns]
        return frame.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_count(symbol):
    url = "https://news.google.com/rss/search?q=" + requests.utils.quote(f'"{symbol}" NSE stock') + "+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return min(len(root.findall(".//item")), 10)
    except Exception:
        return 0

def make_chart(symbol, frame, resistance, support):
    x = frame.tail(140)
    fig = go.Figure(data=[go.Candlestick(x=x.index, open=x.Open, high=x.High, low=x.Low, close=x.Close, name=symbol)])
    fig.add_hline(y=resistance, line_dash="dash", line_color="#e09b35", annotation_text="55-session resistance")
    fig.add_hline(y=support, line_dash="dot", line_color="#269b72", annotation_text="55-session support")
    fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10),
                      yaxis_title="Adjusted price", xaxis_title="Session")
    return fig

def clean_results(frame):
    return frame.drop(columns=["_history"], errors="ignore")

st.title("📈 NSE Swing Breakout Scanner")
st.caption("End-of-day · Daily candles · 2-year history · 10-factor technical score · Screening tool, not investment advice")

with st.sidebar:
    st.header("Daily scan settings")
    universe_mode = st.radio("Universe", ["All NSE EQ stocks", "Custom symbols"], index=0)
    custom_symbols = st.text_area("Symbols (comma separated)", "RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY") if universe_mode == "Custom symbols" else ""
    max_stocks = st.slider("Maximum NSE stocks to request", 100, 2500, 500, 100, disabled=universe_mode == "Custom symbols")
    volume_threshold = st.slider("Breakout volume threshold", 1.2, 3.0, 1.5, 0.1)
    min_score = st.slider("Shortlist minimum score", 0, 100, 65, 5)
    check_news = st.checkbox("Check news for top 40 technical candidates", value=False)
    st.caption("Run after the NSE cash-market close and after the daily candle is available from the data provider.")
    scan_button = st.button("🔍 Run end-of-day scan", type="primary", use_container_width=True)

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if scan_button:
    try:
        with st.spinner("Preparing NSE equity universe..."):
            if universe_mode == "Custom symbols":
                symbols = sorted(set(x.strip().upper().replace(".NS", "") for x in custom_symbols.split(",") if x.strip()))
            else:
                symbols = get_nse_universe()[:max_stocks]
        if not symbols:
            st.error("No symbols found. Check the custom list or NSE symbol source.")
            st.stop()
        progress = st.progress(0, text=f"Downloading 2 years of daily data for {len(symbols)} symbols...")
        histories = download_daily_history(tuple(symbols))
        progress.progress(0.58, text="Checking NIFTY and mapped sector indices...")
        nifty = download_index_data("^NSEI")
        nifty_score = index_strength(nifty)
        sector_indices = sorted(set(SECTOR_MAP.values()))
        sector_frames = {idx: download_index_data(idx) for idx in sector_indices}
        sector_scores = {idx: index_strength(frame) for idx, frame in sector_frames.items()}
        progress.progress(0.68, text=f"Scoring {len(histories)} stocks across the 10 factors...")
        rows = []
        total_histories = max(len(histories), 1)
        for i, (symbol, history) in enumerate(histories.items(), start=1):
            idx = SECTOR_MAP.get(symbol)
            row = analyse_stock(symbol, history, nifty_score, sector_scores.get(idx) if idx else None,
                                catalyst_count=None, volume_threshold=volume_threshold)
            if row:
                rows.append(row)
            if i % 20 == 0 or i == total_histories:
                progress.progress(0.68 + 0.27 * i / total_histories, text=f"Analysed {i}/{len(histories)} usable stocks...")
        if not rows:
            progress.empty()
            st.error("No usable histories were returned. Try a smaller universe or retry later.")
            st.stop()
        results = pd.DataFrame(rows).sort_values(["Score", "Distance to resistance %"], ascending=[False, True]).reset_index(drop=True)
        if check_news:
            progress.progress(0.96, text="Checking recent headlines for the top 40 technical candidates...")
            candidates = results.head(40)["Symbol"].tolist()
            news_map = {sym: get_news_count(sym) for sym in candidates}
            for idx, row in results.iterrows():
                if row["Symbol"] in news_map:
                    history = row["_history"]
                    updated = analyse_stock(row["Symbol"], history, nifty_score,
                                            sector_scores.get(SECTOR_MAP.get(row["Symbol"])) if SECTOR_MAP.get(row["Symbol"]) else None,
                                            catalyst_count=news_map[row["Symbol"]], volume_threshold=volume_threshold)
                    if updated:
                        results.loc[idx, "Catalyst score"] = updated["Catalyst score"]
                        results.loc[idx, "Score"] = updated["Score"]
                        results.loc[idx, "News headlines"] = updated["News headlines"]
            results = results.sort_values("Score", ascending=False).reset_index(drop=True)
        progress.empty()
        st.session_state.scan_results = results
        st.session_state.scan_time = datetime.now().strftime("%d %b %Y, %H:%M IST")
        st.session_state.scan_count = len(symbols)
        st.session_state.history_count = len(histories)
        st.session_state.nifty_score = nifty_score
        st.session_state.latest_bar = max((str(x) for x in results["Latest bar"].dropna()), default="Unknown")
    except Exception as exc:
        st.error(f"Scan failed: {exc}")

results = st.session_state.scan_results
if results is not None:
    threshold_results = results[results.Score >= min_score].copy()
    strong = int((results.Score >= 80).sum())
    confirmed = int((results.Signal == "Confirmed breakout").sum())
    near = int((results.Signal == "Near breakout").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks analysed", f"{len(results):,}")
    c2.metric("Strong setups (80+)", strong)
    c3.metric("Confirmed breakouts", confirmed)
    c4.metric("Near resistance", near)
    st.caption(f"Scan completed: {st.session_state.scan_time} · Latest daily candle: {st.session_state.latest_bar} · Requested universe: {st.session_state.scan_count} · Histories received: {st.session_state.history_count}")

    tabs = st.tabs(["Swing shortlist", "All stocks", "Chart & factor audit", "Scoring guide"])
    with tabs[0]:
        shortlist = threshold_results[threshold_results.Signal != "Not ready"].copy()
        display = clean_results(shortlist)
        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn("Readiness / 100", min_value=0, max_value=100, format="%.1f"),
                                    "Distance to resistance %": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Price": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Resistance": st.column_config.NumberColumn(format="₹%.2f")})
        st.download_button("Download swing shortlist CSV", display.to_csv(index=False), "nse_swing_breakout_shortlist.csv", "text/csv")
        if shortlist.empty:
            st.info("No stocks meet the current score and signal filters. Try lowering the minimum score or review all stocks.")
    with tabs[1]:
        all_display = clean_results(results)
        st.dataframe(all_display, use_container_width=True, hide_index=True)
        st.download_button("Download full scan CSV", all_display.to_csv(index=False), "nse_swing_breakout_all.csv", "text/csv")
    with tabs[2]:
        symbols_in_results = results.Symbol.tolist()
        selected = st.selectbox("Select an NSE symbol", symbols_in_results)
        selected_row = results[results.Symbol == selected].iloc[0]
        st.subheader(f"{selected} · {selected_row['Signal']}")
        a, b, c = st.columns(3)
        a.metric("Readiness score", f"{selected_row['Score']}/100")
        b.metric("Distance to resistance", f"{selected_row['Distance to resistance %']}%")
        c.metric("Relative volume", f"{selected_row['Volume x']}×")
        st.plotly_chart(make_chart(selected, selected_row["_history"], selected_row["Resistance"], selected_row["Support"]), use_container_width=True)
        factor_table = pd.DataFrame([{"Factor": factor, "Score": selected_row[f"{factor} score"], "Maximum": maximum}
                                     for factor, maximum in WEIGHTS.items()])
        st.dataframe(factor_table, use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn("Points", min_value=0, max_value=15, format="%.1f")})
        st.caption(f"Pattern: {selected_row['Pattern']} · Latest bar: {selected_row['Latest bar']} · Sector mapping: {selected_row['Sector index']}")
    with tabs[3]:
        st.markdown("""
### 10-factor scoring (100 points)

| Factor | Maximum | What it checks |
|---|---:|---|
| Volume confirmation | 15 | Latest daily volume versus the preceding 20-session average; surge is rewarded most on a breakout day |
| Level strength | 10 | Separated tests near the prior 55-session resistance |
| Consolidation quality | 10 | Tightness of the last 20 sessions versus the preceding 40 sessions |
| Candle close | 10 | Close location within the day's high-low range |
| Prior trend | 10 | Price versus rising 20/50/200-day averages |
| Market & sector | 10 | NIFTY 50 and mapped NSE sector-index trend; unmapped sectors receive neutral credit |
| Momentum | 10 | RSI(14), MACD histogram and moving-average alignment |
| Catalyst | 5 | Optional count of recent Google News RSS headlines; not sentiment analysis |
| Retest behaviour | 10 | Whether a prior resistance break is holding on a pullback |
| ATR & gap | 10 | ATR expansion and opening-gap quality; very large gaps are treated as chase risk |

### Signal definitions
- **Confirmed breakout:** close above the 55-session resistance buffer, volume at/above the selected threshold and strong candle close.
- **Successful retest candidate:** price has broken the level and is holding a pullback near it.
- **Near breakout:** closing price is within 3% below resistance.
- **Developing setup:** score at least 65 but no immediate breakout/retest condition.
- **Not ready:** none of the above.

### Read this before using the scanner
- The score is a weighted rule-based ranking, **not a probability of profit** and not a buy/sell recommendation.
- Daily bars are fetched from Yahoo Finance. Data may be delayed, incomplete, adjusted, rate-limited or unavailable for some NSE symbols.
- Run after 3:30 PM IST and only after the provider has published the completed daily candle. The app does not guarantee a scheduled run or live alerts.
- Sector mapping is intentionally limited to common large-cap symbols. Unknown sector assignments are shown as unmapped and scored neutrally.
- Pattern names are heuristic candidates; visually inspect each chart. No automated detector can reliably identify every chart formation.
- Backtest the exact rules and paper-trade before using real capital. Include slippage, brokerage, taxes and liquidity in any test.
""")
else:
    st.info("Choose your NSE universe and click **Run end-of-day scan**. The scanner uses daily candles for swing setups, not intraday signals.")
    st.markdown("**Suggested routine:** run after the market closes, review confirmed breakouts first, then near-breakout and developing setups. Open each chart and verify volume, sector context and the closing candle before making any decision.")

st.divider()
st.caption("NSE Swing Breakout Scanner v2.0 · Educational research tool · Data source: Yahoo Finance / NSE symbol list")
