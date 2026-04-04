"""
02_fetch_sentiment.py

通过 EODHD /api/sentiments 端点批量获取 M7 公司的日度聚合情绪数据。
每个 ticker 一次查询，覆盖整个项目所需日期范围。

输出:
  data/sentiment/daily_sentiment.csv   - 日度聚合情绪 (ticker, date, normalized, count)
"""

import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("eodhd_api_key")
if not API_KEY:
    raise ValueError("请在 .env 中设置 eodhd_api_key")

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# 覆盖所有窗口: 最早 ED=2022-01-25, ED-150 ≈ 2021-07-20
# 留足余量，与股价数据对齐
DATE_FROM = "2021-06-01"
DATE_TO = "2025-03-15"

SENTIMENT_DIR = os.path.join(BASE_DIR, "data", "sentiment")
os.makedirs(SENTIMENT_DIR, exist_ok=True)


def fetch_sentiment(ticker: str) -> list[dict]:
    """获取单个 ticker 的日度聚合情绪"""
    url = "https://eodhd.com/api/sentiments"
    params = {
        "s": f"{ticker.lower()}.us",
        "from": DATE_FROM,
        "to": DATE_TO,
        "api_token": API_KEY,
        "fmt": "json",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    key = f"{ticker}.US"
    if key not in data:
        # 尝试小写
        key = f"{ticker.lower()}.us"
    if key not in data:
        # 尝试直接拿第一个key
        keys = list(data.keys())
        if keys:
            key = keys[0]
        else:
            print(f"  ⚠ {ticker}: 返回为空")
            return []

    records = []
    for entry in data[key]:
        records.append({
            "ticker": ticker,
            "date": entry["date"],
            "sentiment": entry["normalized"],
            "news_count": entry["count"],
        })

    return records


def main():
    print("=" * 60)
    print("EODHD 日度情绪数据采集")
    print(f"范围: {DATE_FROM} ~ {DATE_TO}")
    print("=" * 60)

    all_records = []

    for ticker in MAG7:
        print(f"\n📡 {ticker}...", end=" ")
        try:
            records = fetch_sentiment(ticker)
            all_records.extend(records)
            dates = [r["date"] for r in records]
            if dates:
                print(f"✓ {len(records)} 天 ({min(dates)} ~ {max(dates)})")
            else:
                print("⚠ 无数据")
        except Exception as e:
            print(f"✗ 错误: {e}")
        time.sleep(0.5)  # 礼貌性延迟

    # 保存
    df = pd.DataFrame(all_records)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    out_path = os.path.join(SENTIMENT_DIR, "daily_sentiment.csv")
    df.to_csv(out_path, index=False)

    # ── 统计 ──
    print(f"\n{'=' * 60}")
    print("采集结果")
    print(f"{'=' * 60}")
    print(f"总记录: {len(df)} 条")
    print(f"\n每家公司:")
    for ticker in MAG7:
        tk = df[df["ticker"] == ticker]
        if tk.empty:
            print(f"  {ticker:6s}: 无数据")
            continue
        print(f"  {ticker:6s}: {len(tk):>5d} 天  "
              f"({tk['date'].min()} ~ {tk['date'].max()})  "
              f"sentiment 均值: {tk['sentiment'].mean():.3f}  "
              f"日均新闻: {tk['news_count'].mean():.1f} 篇")

    # ── 检查项目所需日期的覆盖率 ──
    print(f"\n── 财报窗口覆盖检查 ──")
    earnings = pd.read_csv(os.path.join(BASE_DIR, "data", "earnings", "mag7_earnings_dates.csv"))

    total_needed = 0
    total_covered = 0

    for _, evt in earnings.iterrows():
        ticker = evt["ticker"]
        ed = pd.to_datetime(evt["earnings_date"])

        tk_dates = set(df[df["ticker"] == ticker]["date"].values)

        # 检查情绪窗口 [ED-7, ED-1]
        for offset in range(1, 8):
            day = (ed - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            total_needed += 1
            if day in tk_dates:
                total_covered += 1

        # 检查静默期 [ED-37, ED-30]
        for offset in range(30, 38):
            day = (ed - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            total_needed += 1
            if day in tk_dates:
                total_covered += 1

    coverage_pct = total_covered / total_needed * 100 if total_needed > 0 else 0
    print(f"  需要的 (ticker, date) 对: {total_needed}")
    print(f"  有数据的: {total_covered} ({coverage_pct:.1f}%)")
    print(f"  缺失的: {total_needed - total_covered}")

    if coverage_pct < 80:
        print("\n  ⚠ 覆盖率较低! 部分日期可能是周末/节假日(无新闻属正常)。")
        print("  建议检查缺失是否集中在周末。")

    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
