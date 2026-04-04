"""
02_fetch_news.py

根据 full_news_schedule.csv 的采集计划，
通过 Finnhub API 抓取新闻并按公司/年份/季度组织存储。

存储结构:
  data/news/{TICKER}/{YEAR}/Q{N}/{TICKER}_{DATE}.json

每个JSON文件包含该公司当天的所有新闻条目。
"""

import finnhub
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict

# ── 加载配置 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    raise ValueError("请在 .env 中设置 FINNHUB_API_KEY")

client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ── 路径 ──
SCHEDULE_PATH = os.path.join(BASE_DIR, "data", "news_schedule", "full_news_schedule.csv")
NEWS_DIR = os.path.join(BASE_DIR, "data", "news")
PROGRESS_PATH = os.path.join(BASE_DIR, "data", "news_schedule", "fetch_progress.json")


def date_to_quarter(date_str: str) -> int:
    """日期 -> 季度号"""
    month = int(date_str.split("-")[1])
    return (month - 1) // 3 + 1


def get_output_path(ticker: str, date_str: str) -> str:
    """生成存储路径: data/news/{TICKER}/{YEAR}/Q{N}/{TICKER}_{DATE}.json"""
    year = date_str[:4]
    quarter = date_to_quarter(date_str)
    dir_path = os.path.join(NEWS_DIR, ticker, year, f"Q{quarter}")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{ticker}_{date_str}.json")


def load_progress() -> set:
    """加载已完成的 (ticker, date) 对，支持断点续传"""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_progress(done: set):
    """保存进度"""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done), f)


def fetch_company_news(ticker: str, date_str: str) -> list[dict]:
    """
    调用 Finnhub company_news API 获取某公司某天的新闻。
    Finnhub 的 company_news 接口接受日期范围，这里用单天。
    """
    try:
        news = client.company_news(ticker, _from=date_str, to=date_str)
        # 只保留需要的字段，减少存储
        cleaned = []
        for article in news:
            cleaned.append({
                "datetime": article.get("datetime"),
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "category": article.get("category", ""),
            })
        return cleaned
    except finnhub.FinnhubAPIException as e:
        print(f"  API错误 {ticker} {date_str}: {e}")
        return []
    except Exception as e:
        print(f"  异常 {ticker} {date_str}: {e}")
        return []


def build_fetch_summary(schedule: pd.DataFrame, earnings_df: pd.DataFrame):
    """
    生成按公司/财报事件组织的抓取时间点汇总。
    """
    print("=" * 70)
    print("抓取时间点汇总（按公司 → 财报事件 → 窗口类型）")
    print("=" * 70)

    tickers = schedule["ticker"].unique()

    for ticker in sorted(tickers):
        tk_schedule = schedule[schedule["ticker"] == ticker]
        tk_earnings = earnings_df[earnings_df["ticker"] == ticker]

        # 该公司作为"主体"关联的财报事件
        own_events = tk_earnings["earnings_date"].unique()

        print(f"\n{'─' * 70}")
        print(f"  {ticker}")
        print(f"{'─' * 70}")

        # 按关联的财报事件分组展示
        for ed in sorted(own_events):
            year = ed[:4]
            q = date_to_quarter(ed)
            print(f"\n  财报: {ed} ({year} Q{q})")

            # 这个公司自身的情绪窗口和静默期
            self_sent = tk_schedule[
                (tk_schedule["window_type"] == "sentiment") &
                (tk_schedule["earnings_date"] == ed)
            ]["collect_date"].sort_values()

            self_quiet = tk_schedule[
                (tk_schedule["window_type"] == "quiet") &
                (tk_schedule["earnings_date"] == ed)
            ]["collect_date"].sort_values()

            if not self_sent.empty:
                print(f"    情绪窗口:  {self_sent.iloc[0]} ~ {self_sent.iloc[-1]}  ({len(self_sent)}天)")
            if not self_quiet.empty:
                print(f"    静默期:    {self_quiet.iloc[0]} ~ {self_quiet.iloc[-1]}  ({len(self_quiet)}天)")

    # 总量统计
    unique_pairs = schedule.drop_duplicates(subset=["ticker", "collect_date"])
    print(f"\n{'=' * 70}")
    print(f"总计独立 (公司, 日期) 抓取任务: {len(unique_pairs)}")
    print(f"{'=' * 70}")


def main():
    # ── 加载采集计划 ──
    schedule = pd.read_csv(SCHEDULE_PATH)
    earnings_df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "earnings", "mag7_earnings_dates.csv")
    )

    # ── 去重：按 (ticker, collect_date) 只需要抓一次 ──
    unique_tasks = (
        schedule[["ticker", "collect_date"]]
        .drop_duplicates()
        .sort_values(["ticker", "collect_date"])
        .reset_index(drop=True)
    )

    # ── 打印抓取汇总 ──
    build_fetch_summary(schedule, earnings_df)

    # ── 开始抓取 ──
    done = load_progress()
    total = len(unique_tasks)
    already_done = sum(1 for _, r in unique_tasks.iterrows()
                       if (r["ticker"], r["collect_date"]) in done)

    print(f"\n已完成: {already_done}/{total}，剩余: {total - already_done}")

    if already_done == total:
        print("全部已完成！无需继续抓取。")
        return

    if "--no-confirm" not in sys.argv:
        input_msg = input(f"\n确认开始抓取 {total - already_done} 条？(y/n): ")
        if input_msg.lower() != "y":
            print("已取消。")
            return

    # ── 按 ticker 分组抓取，方便追踪进度 ──
    stats = defaultdict(lambda: {"fetched": 0, "articles": 0, "empty": 0, "errors": 0})
    call_count = 0
    start_time = time.time()

    for idx, row in unique_tasks.iterrows():
        ticker = row["ticker"]
        date_str = row["collect_date"]
        key = (ticker, date_str)

        if key in done:
            continue

        # ── 速率控制: Finnhub 免费 60次/min ──
        call_count += 1
        if call_count % 58 == 0:
            elapsed = time.time() - start_time
            if elapsed < 62:
                wait = 62 - elapsed
                print(f"  ⏳ 速率限制，等待 {wait:.0f}s...")
                time.sleep(wait)
            start_time = time.time()

        # ── 抓取 ──
        output_path = get_output_path(ticker, date_str)

        # 如果文件已存在（之前抓过但progress没记录），跳过
        if os.path.exists(output_path):
            done.add(key)
            continue

        articles = fetch_company_news(ticker, date_str)

        # ── 存储 ──
        result = {
            "ticker": ticker,
            "date": date_str,
            "fetch_time": datetime.now().isoformat(),
            "article_count": len(articles),
            "articles": articles,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # ── 更新统计 ──
        done.add(key)
        stats[ticker]["fetched"] += 1
        stats[ticker]["articles"] += len(articles)
        if len(articles) == 0:
            stats[ticker]["empty"] += 1

        # ── 进度输出（每50条） ──
        completed = len(done)
        if completed % 50 == 0 or completed == total:
            pct = completed / total * 100
            print(f"  [{completed}/{total}] ({pct:.1f}%) "
                  f"{ticker} {date_str} → {len(articles)} 篇")
            save_progress(done)

    # ── 最终保存 ──
    save_progress(done)

    # ── 抓取报告 ──
    print(f"\n{'=' * 70}")
    print("抓取完成！统计：")
    print(f"{'=' * 70}")
    print(f"{'公司':8s} {'抓取天数':>8s} {'新闻总数':>8s} {'空天数':>6s}")
    print(f"{'-' * 36}")

    total_articles = 0
    for ticker in sorted(stats.keys()):
        s = stats[ticker]
        total_articles += s["articles"]
        print(f"{ticker:8s} {s['fetched']:>8d} {s['articles']:>8d} {s['empty']:>6d}")

    print(f"\n新闻总数: {total_articles}")
    print(f"存储目录: {NEWS_DIR}")


if __name__ == "__main__":
    main()
