"""
clean_data.py

Clean all CSV data files, standardize date formats, remove junk columns,
then rebuild the SQLite database.

Issues fixed:
  - window_articles.article_date: mixed formats → YYYY-MM-DD
  - window_articles.earnings_date: MM-DD-YY → YYYY-MM-DD
  - window_articles.source: all NaN → dropped
  - window_articles.polarity: 19 NaN rows → dropped
  - daily_sentiment: no issues, just verify
  - daily_prices: no issues, just verify
  - earnings: no issues, just verify
"""

import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "hackathon.db")


def clean_window_articles():
    path = os.path.join(BASE_DIR, "data/news/window_articles.csv")
    df = pd.read_csv(path)
    print(f"window_articles: {len(df)} rows loaded")

    # 1. Drop source column (all NaN)
    df = df.drop(columns=["source"])

    # 2. Standardize earnings_date: MM-DD-YY → YYYY-MM-DD
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], format="mixed").dt.strftime("%Y-%m-%d")

    # 3. Standardize article_date → YYYY-MM-DD (drop time component)
    #    Two formats: "M/DD/YY HH:MM" and "MM-DD-YY"
    df["article_date"] = pd.to_datetime(df["article_date"], format="mixed").dt.strftime("%Y-%m-%d")

    # 4. Drop rows with null polarity
    null_count = df["polarity"].isna().sum()
    df = df.dropna(subset=["polarity"])
    print(f"  Dropped {null_count} rows with null polarity")

    # 5. Verify
    assert df["earnings_date"].str.match(r"\d{4}-\d{2}-\d{2}").all(), "earnings_date format error"
    assert df["article_date"].str.match(r"\d{4}-\d{2}-\d{2}").all(), "article_date format error"
    assert df["polarity"].notna().all(), "polarity still has nulls"

    # 6. Sort
    df = df.sort_values(["ticker", "earnings_date", "article_date"]).reset_index(drop=True)

    df.to_csv(path, index=False)
    print(f"  Saved: {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def verify_other_tables():
    """Quick sanity check on other CSVs."""
    # earnings
    e = pd.read_csv(os.path.join(BASE_DIR, "data/earnings/mag7_earnings.csv"))
    assert e["earnings_date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert e["surprise_pct"].notna().all()
    print(f"earnings: OK ({len(e)} rows)")

    # daily_prices
    p = pd.read_csv(os.path.join(BASE_DIR, "data/prices/daily_prices.csv"))
    assert p["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    print(f"daily_prices: OK ({len(p)} rows)")

    # daily_sentiment
    s = pd.read_csv(os.path.join(BASE_DIR, "data/sentiment/daily_sentiment.csv"))
    assert s["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    print(f"daily_sentiment: OK ({len(s)} rows)")


def rebuild_db():
    """Rebuild SQLite from cleaned CSVs."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    tables = {
        "earnings": "data/earnings/mag7_earnings.csv",
        "daily_prices": "data/prices/daily_prices.csv",
        "daily_sentiment": "data/sentiment/daily_sentiment.csv",
        "window_articles": "data/news/window_articles.csv",
        "extreme_events": "data/sentiment/extreme_events.csv",
    }

    for name, path in tables.items():
        df = pd.read_csv(os.path.join(BASE_DIR, path))
        df.to_sql(name, conn, index=False)
        print(f"  {name}: {len(df)} rows")

    # Indexes
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
    conn.close()

    size = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\n  DB rebuilt: {size:.1f} MB")


def main():
    print("=" * 50)
    print("Data Cleaning")
    print("=" * 50)

    print("\n── Cleaning window_articles ──")
    clean_window_articles()

    print("\n── Verifying other tables ──")
    verify_other_tables()

    print("\n── Rebuilding database ──")
    rebuild_db()

    # Final verification
    print("\n── Final verification ──")
    conn = sqlite3.connect(DB_PATH)
    for table in ["earnings", "daily_prices", "daily_sentiment", "window_articles"]:
        row = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
        cols = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 1").description]
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:20s}: {count:>6d} rows  cols={cols}")

    # Spot check date formats in window_articles
    sample = conn.execute(
        "SELECT ticker, earnings_date, article_date, polarity FROM window_articles LIMIT 3"
    ).fetchall()
    print(f"\n  window_articles sample:")
    for r in sample:
        print(f"    {r}")
    conn.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
