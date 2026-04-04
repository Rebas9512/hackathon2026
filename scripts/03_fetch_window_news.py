"""
04_fetch_window_news.py

为每个财报事件抓取 [ED-7, ED-1] 窗口期的所有新闻及逐篇情绪评分。

输出:
  data/news/window_articles.csv  - 所有窗口期新闻 (逐篇, 含 polarity/neg/neu/pos)
"""

import requests
import pandas as pd
import os
import time
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("eodhd_api_key")
if not API_KEY:
    raise ValueError("请在 .env 中设置 eodhd_api_key")

NEWS_DIR = os.path.join(BASE_DIR, "data", "news")
os.makedirs(NEWS_DIR, exist_ok=True)


def fetch_news(ticker: str, date_from: str, date_to: str) -> list[dict]:
    """获取指定 ticker 在日期范围内的所有新闻"""
    all_articles = []
    offset = 0
    limit = 100

    while True:
        url = "https://eodhd.com/api/news"
        params = {
            "s": f"{ticker}.US",
            "from": date_from,
            "to": date_to,
            "limit": limit,
            "offset": offset,
            "api_token": API_KEY,
            "fmt": "json",
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"    API error {resp.status_code}: {resp.text[:200]}")
            break

        articles = resp.json()
        if not articles:
            break

        all_articles.extend(articles)

        if len(articles) < limit:
            break
        offset += limit
        time.sleep(0.2)

    return all_articles


def main():
    earnings = pd.read_csv(
        os.path.join(BASE_DIR, "data", "earnings", "mag7_earnings_dates.csv")
    )

    print("=" * 60)
    print("抓取财报前 7 天窗口期新闻 (逐篇)")
    print(f"共 {len(earnings)} 个财报事件")
    print("=" * 60)

    all_records = []
    api_calls = 0

    for idx, evt in earnings.iterrows():
        ticker = evt["ticker"]
        ed = pd.to_datetime(evt["earnings_date"])

        # [ED-7, ED-1]
        date_from = (ed - timedelta(days=7)).strftime("%Y-%m-%d")
        date_to = (ed - timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"  [{idx+1:>2d}/91] {ticker} ED={evt['earnings_date']}  "
              f"window={date_from}~{date_to}", end="")

        articles = fetch_news(ticker, date_from, date_to)
        api_calls += 1 + (len(articles) // 100)  # 估算分页次数

        for art in articles:
            sent = art.get("sentiment", {}) or {}
            all_records.append({
                "ticker": ticker,
                "earnings_date": evt["earnings_date"],
                "article_date": art.get("date", ""),
                "title": art.get("title", ""),
                "polarity": sent.get("polarity"),
                "neg": sent.get("neg"),
                "neu": sent.get("neu"),
                "pos": sent.get("pos"),
                "source": art.get("source", ""),
            })

        print(f"  → {len(articles)} 篇")
        time.sleep(0.3)

    # 保存
    df = pd.DataFrame(all_records)
    out_path = os.path.join(NEWS_DIR, "window_articles.csv")
    df.to_csv(out_path, index=False)

    # 统计
    print(f"\n{'=' * 60}")
    print("采集结果")
    print(f"{'=' * 60}")
    print(f"总文章数: {len(df)}")
    print(f"API 调用 (估算): ~{api_calls * 5} calls")

    print(f"\n每家公司:")
    for ticker in sorted(df["ticker"].unique()):
        tk = df[df["ticker"] == ticker]
        events = tk["earnings_date"].nunique()
        avg = len(tk) / events if events > 0 else 0
        pol = tk["polarity"].dropna()
        print(f"  {ticker:6s}: {len(tk):>5d} 篇 / {events} 事件  "
              f"(均 {avg:.0f} 篇/事件)  "
              f"polarity 均值={pol.mean():.3f}  std={pol.std():.3f}")

    # 检查是否有事件缺新闻
    events_with_news = df.groupby(["ticker", "earnings_date"]).size().reset_index(name="count")
    all_events = earnings[["ticker", "earnings_date"]].copy()
    merged = all_events.merge(events_with_news, on=["ticker", "earnings_date"], how="left")
    empty = merged[merged["count"].isna() | (merged["count"] == 0)]

    if not empty.empty:
        print(f"\n⚠ {len(empty)} 个事件无新闻:")
        for _, r in empty.iterrows():
            print(f"  {r['ticker']} {r['earnings_date']}")
    else:
        print(f"\n✓ 所有 91 个事件都有窗口期新闻")

    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
