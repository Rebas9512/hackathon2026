"""
build_db.py

将所有 CSV 数据导入 SQLite 数据库，供后续脚本统一调用。

输出: data/hackathon.db
"""

import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "hackathon.db")


def build():
    # 删除旧库重建
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    # ── 1. earnings ──
    df = pd.read_csv(os.path.join(BASE_DIR, "data/earnings/mag7_earnings.csv"))
    df.to_sql("earnings", conn, index=False)
    print(f"earnings: {len(df)} rows")

    # ── 2. daily_prices ──
    df = pd.read_csv(os.path.join(BASE_DIR, "data/prices/daily_prices.csv"))
    df.to_sql("daily_prices", conn, index=False)
    print(f"daily_prices: {len(df)} rows")

    # ── 3. daily_sentiment ──
    df = pd.read_csv(os.path.join(BASE_DIR, "data/sentiment/daily_sentiment.csv"))
    df.to_sql("daily_sentiment", conn, index=False)
    print(f"daily_sentiment: {len(df)} rows")

    # ── 4. window_articles ──
    df = pd.read_csv(os.path.join(BASE_DIR, "data/news/window_articles.csv"))
    df.to_sql("window_articles", conn, index=False)
    print(f"window_articles: {len(df)} rows")

    # ── 5. extreme_events ──
    df = pd.read_csv(os.path.join(BASE_DIR, "data/sentiment/extreme_events.csv"))
    df.to_sql("extreme_events", conn, index=False)
    print(f"extreme_events: {len(df)} rows")

    # ── 索引 ──
    cur = conn.cursor()
    indexes = [
        "CREATE INDEX idx_prices_ticker_date ON daily_prices(ticker, date)",
        "CREATE INDEX idx_sentiment_ticker_date ON daily_sentiment(ticker, date)",
        "CREATE INDEX idx_articles_ticker_ed ON window_articles(ticker, earnings_date)",
        "CREATE INDEX idx_earnings_ticker ON earnings(ticker, earnings_date)",
    ]
    for sql in indexes:
        cur.execute(sql)
    conn.commit()
    print("\nIndexes created.")

    # ── 验证 ──
    print(f"\n{'=' * 50}")
    print("验证")
    print(f"{'=' * 50}")
    for table in ["earnings", "daily_prices", "daily_sentiment", "window_articles", "extreme_events"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        print(f"  {table:20s}: {count:>6d} rows  cols={cols}")

    db_size = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\nDB size: {db_size:.1f} MB")
    print(f"Saved: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    build()
