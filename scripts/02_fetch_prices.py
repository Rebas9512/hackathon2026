"""
03_fetch_prices.py

获取 M7 + SPY 的日度 OHLCV 数据和财报 EPS surprise 数据。
日期范围覆盖最早财报事件的 ED-150 到最晚财报事件的 ED+10。

输出:
  data/prices/daily_prices.csv      - 日度 OHLCV
  data/earnings/mag7_earnings.csv   - 财报日期 + EPS surprise
"""

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# ── 配置 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_DIR = os.path.join(BASE_DIR, "data", "prices")
EARNINGS_DIR = os.path.join(BASE_DIR, "data", "earnings")

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
BENCHMARK = ["SPY"]
ALL_TICKERS = MAG7 + BENCHMARK

# 日期范围: 留足余量
# 最早 ED: 2022-01-25, ED-150 ≈ 2021-07-01
# 最晚 ED: 2025-02-26, ED+10 ≈ 2025-03-10
PRICE_START = "2021-06-01"
PRICE_END = "2025-03-31"


def fetch_prices():
    """批量下载日度 OHLCV"""
    print("=" * 60)
    print("下载日度股价数据")
    print(f"Tickers: {ALL_TICKERS}")
    print(f"Range: {PRICE_START} ~ {PRICE_END}")
    print("=" * 60)

    df = yf.download(
        tickers=ALL_TICKERS,
        start=PRICE_START,
        end=PRICE_END,
        auto_adjust=True,
        threads=True,
    )

    # yf.download 返回 MultiIndex columns: (field, ticker)
    # 转成长格式方便后续使用
    records = []
    for ticker in ALL_TICKERS:
        try:
            ticker_df = pd.DataFrame({
                "date": df.index,
                "ticker": ticker,
                "open": df[("Open", ticker)].values,
                "high": df[("High", ticker)].values,
                "low": df[("Low", ticker)].values,
                "close": df[("Close", ticker)].values,
                "volume": df[("Volume", ticker)].values,
            })
            records.append(ticker_df)
            print(f"  ✓ {ticker}: {ticker_df['close'].notna().sum()} 交易日")
        except KeyError:
            print(f"  ✗ {ticker}: 数据缺失")

    prices = pd.concat(records, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")

    os.makedirs(PRICES_DIR, exist_ok=True)
    out_path = os.path.join(PRICES_DIR, "daily_prices.csv")
    prices.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")
    print(f"总行数: {len(prices)}")
    return prices


def fetch_earnings_surprise():
    """获取 EPS surprise 数据并与已有财报日期合并"""
    print("\n" + "=" * 60)
    print("获取 EPS Surprise 数据")
    print("=" * 60)

    # 加载已有的财报日期
    dates_path = os.path.join(EARNINGS_DIR, "mag7_earnings_dates.csv")
    existing = pd.read_csv(dates_path)
    existing["earnings_date"] = pd.to_datetime(existing["earnings_date"])
    print(f"已有财报事件: {len(existing)} 条")

    all_eps = []
    for ticker in MAG7:
        print(f"\n📡 {ticker}:")
        stock = yf.Ticker(ticker)

        # 获取 earnings_dates (包含 EPS Estimate/Actual/Surprise)
        try:
            ed = stock.get_earnings_dates(limit=60)
            if ed is None or ed.empty:
                print(f"  ⚠ 无 earnings_dates 数据")
                continue

            # 清理
            ed = ed.reset_index()
            ed.columns = [c.strip() for c in ed.columns]

            # 找到日期列
            date_col = ed.columns[0]  # 通常是 'Earnings Date'
            ed["earnings_date"] = pd.to_datetime(ed[date_col])
            if ed["earnings_date"].dt.tz is not None:
                ed["earnings_date"] = ed["earnings_date"].dt.tz_localize(None)

            # 提取需要的列
            eps_cols = {}
            for col in ed.columns:
                col_lower = col.lower()
                if "estimate" in col_lower and "eps" in col_lower:
                    eps_cols["eps_estimate"] = col
                elif "reported" in col_lower or ("actual" in col_lower and "eps" in col_lower):
                    eps_cols["eps_actual"] = col
                elif "surprise" in col_lower and "%" in col_lower:
                    eps_cols["surprise_pct"] = col
                elif "surprise" in col_lower:
                    eps_cols["surprise"] = col

            print(f"  找到列: {list(eps_cols.keys())}")

            for _, row in ed.iterrows():
                record = {
                    "ticker": ticker,
                    "earnings_date": row["earnings_date"],
                }
                for key, col in eps_cols.items():
                    record[key] = row.get(col)
                all_eps.append(record)

            print(f"  ✓ {len(ed)} 条记录")

        except Exception as e:
            print(f"  ✗ 错误: {e}")

    if not all_eps:
        print("\n⚠ 未获取到任何 EPS 数据")
        return None

    eps_df = pd.DataFrame(all_eps)
    eps_df["earnings_date"] = pd.to_datetime(eps_df["earnings_date"])

    # 与已有日期匹配 (允许 ±1 天误差)
    merged_records = []
    for _, evt in existing.iterrows():
        ticker = evt["ticker"]
        ed = evt["earnings_date"]

        # 在 eps_df 中找匹配
        candidates = eps_df[
            (eps_df["ticker"] == ticker) &
            (abs((eps_df["earnings_date"] - ed).dt.days) <= 1)
        ]

        record = {
            "ticker": evt["ticker"],
            "company": evt["company"],
            "earnings_date": ed.strftime("%Y-%m-%d"),
        }

        if not candidates.empty:
            match = candidates.iloc[0]
            for col in ["eps_estimate", "eps_actual", "surprise", "surprise_pct"]:
                if col in match.index:
                    record[col] = match[col]

        merged_records.append(record)

    result = pd.DataFrame(merged_records)

    out_path = os.path.join(EARNINGS_DIR, "mag7_earnings.csv")
    result.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")
    print(f"总事件: {len(result)} 条")

    # 统计 EPS 覆盖率
    if "eps_actual" in result.columns:
        coverage = result["eps_actual"].notna().sum()
        print(f"EPS actual 覆盖: {coverage}/{len(result)} ({coverage/len(result)*100:.1f}%)")
    if "surprise_pct" in result.columns:
        coverage = result["surprise_pct"].notna().sum()
        print(f"Surprise % 覆盖: {coverage}/{len(result)} ({coverage/len(result)*100:.1f}%)")

    return result


def validate(prices, earnings):
    """数据完整性检查"""
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)

    # 1. 股价完整性
    print("\n── 股价数据 ──")
    for ticker in ALL_TICKERS:
        tk = prices[prices["ticker"] == ticker]
        valid = tk["close"].notna().sum()
        total = len(tk)
        date_min = tk["date"].min()
        date_max = tk["date"].max()
        print(f"  {ticker:6s}: {valid:>4d} 交易日  ({date_min} ~ {date_max})"
              f"  缺失: {total - valid}")

    # 2. 检查每个财报事件的数据覆盖
    print("\n── 财报事件覆盖检查 ──")
    issues = []
    if earnings is not None:
        for _, evt in earnings.iterrows():
            ticker = evt["ticker"]
            ed = pd.to_datetime(evt["earnings_date"])

            tk_prices = prices[prices["ticker"] == ticker].copy()
            tk_prices["date"] = pd.to_datetime(tk_prices["date"])
            tk_prices = tk_prices.sort_values("date")

            # 检查 ED-150 之前是否有数据
            window_start = ed - timedelta(days=220)  # 日历日，足够覆盖 150 交易日
            pre_data = tk_prices[(tk_prices["date"] >= window_start) & (tk_prices["date"] < ed)]
            trading_days_before = len(pre_data)

            # 检查 ED+5 是否有数据
            post_data = tk_prices[(tk_prices["date"] > ed) & (tk_prices["date"] <= ed + timedelta(days=10))]
            trading_days_after = len(post_data)

            if trading_days_before < 150:
                issues.append(f"  ⚠ {ticker} {evt['earnings_date']}: ED前只有 {trading_days_before} 交易日 (需要150)")
            if trading_days_after < 5:
                issues.append(f"  ⚠ {ticker} {evt['earnings_date']}: ED后只有 {trading_days_after} 交易日 (需要5)")

    if issues:
        print(f"  发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个
            print(issue)
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个")
    else:
        print("  ✓ 所有财报事件的股价数据覆盖完整")

    # 3. EPS 数据摘要
    if earnings is not None and "eps_actual" in earnings.columns:
        print("\n── EPS 数据摘要 ──")
        for ticker in MAG7:
            tk = earnings[earnings["ticker"] == ticker]
            has_eps = tk["eps_actual"].notna().sum()
            print(f"  {ticker:6s}: {has_eps}/{len(tk)} 季度有 EPS 数据")

        # 缺失的具体事件
        if "eps_actual" in earnings.columns:
            missing = earnings[earnings["eps_actual"].isna()]
            if not missing.empty:
                print(f"\n  缺失 EPS 的事件:")
                for _, row in missing.iterrows():
                    print(f"    {row['ticker']} {row['earnings_date']}")


def main():
    prices = fetch_prices()
    earnings = fetch_earnings_surprise()
    validate(prices, earnings)


if __name__ == "__main__":
    main()
