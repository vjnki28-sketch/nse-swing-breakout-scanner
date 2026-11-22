"""Core calculations for the NSE EOD swing-breakout scanner.

All signals are heuristic screening signals, not forecasts or trade instructions.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

WEIGHTS = {
    "Volume confirmation": 15,
    "Level strength": 10,
    "Consolidation quality": 10,
    "Candle close": 10,
    "Prior trend": 10,
    "Market & sector": 10,
    "Momentum": 10,
    "Catalyst": 5,
    "Retest behaviour": 10,
    "ATR & gap": 10,
}

# Common NSE symbols mapped to NSE sector indices. Unmapped symbols receive
# a neutral sector sub-score; they are never guessed into a sector.
SECTOR_MAP = {}
for _symbols, _index in [
    ("HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK INDUSINDBK BANKBARODA PNB", "^NSEBANK"),
    ("INFY TCS HCLTECH WIPRO TECHM LTIM PERSISTENT COFORGE MPHASIS", "^CNXIT"),
    ("RELIANCE ONGC IOC BPCL HPCL GAIL OIL", "^CNXENERGY"),
    ("ITC HINDUNILVR NESTLEIND BRITANNIA DABUR GODREJCP TATACONSUM", "^CNXFMCG"),
    ("TATASTEEL JSWSTEEL HINDALCO VEDL SAIL NMDC", "^CNXMETAL"),
    ("SUNPHARMA DRREDDY CIPLA DIVISLAB APOLLOHOSP AUROPHARMA", "^CNXPHARMA"),
    ("MARUTI TATAMOTORS M&M HEROMOTOCO EICHERMOT BAJAJ-AUTO", "^CNXAUTO"),
    ("LT ADANIPORTS ULTRACEMCO GRASIM AMBUJACEM SHREECEM", "^CNXINFRA"),
    ("DLF GODREJPROP OBEROIRLTY PRESTIGE", "^CNXREALTY"),
    ("BAJFINANCE BAJAJFINSV CHOLAFIN SHRIRAMFIN SBICARD", "^CNXFIN"),
]:
    for _symbol in _symbols.split():
        SECTOR_MAP[_symbol] = _index


def _series(d: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(d[name], errors="coerce")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(d: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = _series(d, "High"), _series(d, "Low"), _series(d, "Close")
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def macd(close: pd.Series):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def _touch_count(highs: pd.Series, level: float, tolerance: float = 0.015) -> int:
    """Count separate resistance tests, not every adjacent candle in a cluster."""
    hits = np.flatnonzero((highs >= level * (1 - tolerance)) & (highs <= level * (1 + tolerance)))
    if len(hits) == 0:
        return 0
    clusters = 1
    last = hits[0]
    for idx in hits[1:]:
        if idx - last >= 3:
            clusters += 1
        last = idx
    return clusters


def detect_pattern(d: pd.DataFrame) -> str:
    x = d.tail(45).dropna(subset=["High", "Low", "Close"])
    if len(x) < 35:
        return "Insufficient history"
    highs, lows = x.High.to_numpy(float), x.Low.to_numpy(float)
    xx = np.arange(len(x), dtype=float)
    hs = np.polyfit(xx, highs, 1)[0] / max(np.mean(highs), 1e-9)
    ls = np.polyfit(xx, lows, 1)[0] / max(np.mean(lows), 1e-9)
    recent_width = (x.High.tail(15).max() - x.Low.tail(15).min()) / x.Close.tail(15).mean()
    earlier_width = (x.High.iloc[:30].max() - x.Low.iloc[:30].min()) / x.Close.iloc[:30].mean()
    prior_rise = x.Close.iloc[-18] / x.Close.iloc[-35] - 1
    if hs < -0.00045 and ls > 0.00045:
        return "Symmetrical triangle candidate"
    if abs(hs) < 0.0004 and ls > 0.00045:
        return "Ascending triangle candidate"
    if hs < -0.00045 and abs(ls) < 0.0004:
        return "Descending triangle candidate"
    if prior_rise > 0.10 and recent_width < 0.075:
        return "Bull-flag candidate"
    if recent_width < earlier_width * 0.65 and recent_width < 0.14:
        return "Tightening / flat base"
    # Simple double-bottom candidate: two separated troughs within 3%, with a
    # meaningful intervening recovery. This is a visual-screening heuristic.
    segment = x.Low.reset_index(drop=True)
    minima = [i for i in range(2, len(segment)-2)
              if segment.iloc[i] <= segment.iloc[i-1:i+2].min()]
    for i, left in enumerate(minima):
        for right in minima[i+1:]:
            if right - left >= 8:
                a, b = segment.iloc[left], segment.iloc[right]
                between = segment.iloc[left:right+1].max()
                if abs(a-b) / max((a+b)/2, 1e-9) <= 0.03 and between >= max(a,b)*1.05:
                    return "Double-bottom candidate"
    return "No clear textbook pattern"


def index_strength(index_data: Optional[pd.DataFrame]) -> Optional[float]:
    if index_data is None or len(index_data) < 55 or "Close" not in index_data:
        return None
    c = _series(index_data, "Close").dropna()
    if len(c) < 55:
        return None
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    latest = c.iloc[-1]
    score = (2.5 if latest > ma20.iloc[-1] else 0) + (2.5 if ma20.iloc[-1] > ma50.iloc[-1] else 0)
    return score


def analyse_stock(
    symbol: str,
    data: pd.DataFrame,
    nifty_score: Optional[float] = None,
    sector_score: Optional[float] = None,
    catalyst_count: Optional[int] = None,
    volume_threshold: float = 1.5,
) -> Optional[dict]:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if data is None or not required.issubset(data.columns):
        return None
    d = data.copy().dropna(subset=list(required)).sort_index()
    if len(d) < 210:
        return None
    o, h, l, c, v = (_series(d, k) for k in ["Open", "High", "Low", "Close", "Volume"])
    last, prev = d.iloc[-1], d.iloc[-2]
    price = float(c.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    # Use the previous 55 completed sessions, excluding today's candle.
    lookback_high = h.iloc[-56:-1]
    resistance = float(lookback_high.max())
    support = float(l.iloc[-56:-1].min())
    distance_pct = (resistance / price - 1) * 100
    breakout = price > resistance * 1.002
    near = 0 <= distance_pct <= 3.0
    close_position = float(np.clip((c.iloc[-1] - l.iloc[-1]) / max(h.iloc[-1] - l.iloc[-1], 1e-9), 0, 1))

    vol_avg = float(v.rolling(20).mean().iloc[-2])
    vol_ratio = float(v.iloc[-1] / vol_avg) if vol_avg > 0 else 0.0
    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    r = float(rsi(c).iloc[-1])
    m_line, m_signal, m_hist = macd(c)
    atr_s = atr(d)
    atr_now = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else 0.0
    atr_prev = float(atr_s.iloc[-6]) if pd.notna(atr_s.iloc[-6]) else 0.0
    atr_pct = atr_now / price * 100 if price else 0
    atr_expansion = atr_now / atr_prev if atr_prev > 0 else 1.0
    gap_pct = (float(o.iloc[-1]) / float(c.iloc[-2]) - 1) * 100 if c.iloc[-2] else 0.0

    # Tightness of the last 20 sessions vs the preceding 40 sessions.
    recent_range = (h.iloc[-21:-1].max() - l.iloc[-21:-1].min()) / max(c.iloc[-21:-1].mean(), 1e-9)
    prior_range = (h.iloc[-61:-21].max() - l.iloc[-61:-21].min()) / max(c.iloc[-61:-21].mean(), 1e-9)
    touches = _touch_count(h.iloc[-61:-1].reset_index(drop=True), resistance)
    retest = bool(float(prev.Close) > resistance and float(last.Low) <= resistance * 1.02 and price >= resistance)
    recent_retest = bool((c.iloc[-4:-1] > resistance).any() and (l.iloc[-3:] <= resistance * 1.02).any() and price >= resistance)

    # 1. Volume: volume surge matters most on the actual breakout day.
    vol_score = 15 * min(vol_ratio / max(volume_threshold, 0.1), 1.0) if breakout else 15 * min(vol_ratio / 2.5, 0.65)
    # 2. Level quality: repeated, separated tests add significance.
    level_score = min(touches / 4, 1.0) * 10
    # 3. Base tightness and volatility contraction.
    tight_component = 7 if recent_range < 0.10 else 5 if recent_range < 0.15 else 2 if recent_range < 0.22 else 0
    compression_component = 3 if prior_range > 0 and recent_range < prior_range * 0.75 else 0
    consolidation_score = tight_component + compression_component
    # 4. Candle close strength.
    candle_score = 10 * close_position if (breakout or near) else 10 * close_position * 0.5
    # 5. Existing daily trend.
    trend_bits = [price > float(ma50.iloc[-1]), float(ma50.iloc[-1]) > float(ma50.iloc[-6]),
                  price > float(ma200.iloc[-1]), float(ma20.iloc[-1]) > float(ma50.iloc[-1])]
    trend_score = sum(trend_bits) / len(trend_bits) * 10
    # 6. Market and sector; unknown sector is neutral, not falsely strong.
    broad = 2.5 if nifty_score is None else nifty_score
    sector = 2.5 if sector_score is None else sector_score
    context_score = broad + sector
    # 7. RSI, MACD and moving-average momentum.
    momentum_score = (3.0 if 50 <= r <= 70 else 1.5 if 45 <= r < 50 or 70 < r <= 76 else 0)
    momentum_score += 3.5 if m_line.iloc[-1] > m_signal.iloc[-1] and m_hist.iloc[-1] > m_hist.iloc[-2] else 0
    momentum_score += 3.5 if price > ma20.iloc[-1] > ma50.iloc[-1] else 0
    # 8. Catalyst: missing news scan gets neutral credit, not a false negative.
    news_n = catalyst_count
    catalyst_score = 2.5 if news_n is None else min(max(news_n, 0) / 3, 1) * 5
    # 9. Retest behavior; pre-breakout setups receive limited credit only.
    retest_score = 10 if (retest or recent_retest) else 4 if (c.iloc[-4:-1] > resistance).any() else 2 if near else 0
    # 10. ATR expansion is useful; very large gaps are penalised as chase risk.
    atr_component = 6 if atr_expansion >= 1.08 else 4 if atr_expansion >= 0.95 else 2
    gap_abs = abs(gap_pct)
    gap_component = 4 if 0.3 <= gap_pct <= 3.0 else 2 if -1.0 <= gap_pct < 0.3 else 0
    if gap_abs > 5:
        gap_component = 0
    atr_gap_score = atr_component + gap_component

    scores = {
        "Volume confirmation": vol_score,
        "Level strength": level_score,
        "Consolidation quality": consolidation_score,
        "Candle close": candle_score,
        "Prior trend": trend_score,
        "Market & sector": context_score,
        "Momentum": momentum_score,
        "Catalyst": catalyst_score,
        "Retest behaviour": retest_score,
        "ATR & gap": atr_gap_score,
    }
    total = round(sum(scores.values()), 1)
    if breakout and vol_ratio >= volume_threshold and close_position >= 0.65:
        signal = "Confirmed breakout"
    elif retest or recent_retest:
        signal = "Successful retest candidate"
    elif near:
        signal = "Near breakout"
    elif total >= 65:
        signal = "Developing setup"
    else:
        signal = "Not ready"

    sector_index = SECTOR_MAP.get(symbol)
    return {
        "Symbol": symbol,
        "Price": round(price, 2),
        "Resistance": round(resistance, 2),
        "Support": round(support, 2),
        "Distance to resistance %": round(distance_pct, 2),
        "Volume x": round(vol_ratio, 2),
        "RSI": round(r, 1),
        "ATR %": round(atr_pct, 2),
        "ATR expansion x": round(atr_expansion, 2),
        "Pattern": detect_pattern(d),
        "Signal": signal,
        "Score": total,
        "Market/sector": "Measured" if nifty_score is not None and sector_index and sector_score is not None else "Partial / neutral",
        "Sector index": sector_index or "Not mapped",
        "News headlines": news_n if news_n is not None else "Not checked",
        "Latest bar": pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d"),
        **{f"{k} score": round(float(v), 1) for k, v in scores.items()},
        "_history": d,
    }
