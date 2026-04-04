"""
01_build_news_schedule.py

为每个M7公司获取2022-2025年的历史财报日期，
然后计算所有需要新闻采集的时间窗口：
  - 情绪窗口: [ED-7, ED-1]
  - 静默期窗口: [ED-37, ED-30]
输出完整的 (公司, 日期, 窗口类型, 关联财报事件) 采集清单。
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# ── 配置 ──────────────────────────────────────────────
MAG7 = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
}

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 3, 31)  # 只做到2025 Q1

# 时间窗口参数
SENTIMENT_WINDOW = 7   # 财报前7天
QUIET_START = 37       # 静默期起点 ED-37
QUIET_END = 30         # 静默期终点 ED-30

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_earnings_dates(ticker: str) -> list[dict]:
    """从yfinance获取历史财报日期"""
    stock = yf.Ticker(ticker)

    try:
        # yfinance的earnings_dates返回过去和未来的财报日期
        # 需要多次请求以获取足够的历史数据
        all_dates = []
        # 尝试获取尽可能多的历史财报日期
        for limit in [20, 40, 60]:
            try:
                ed = stock.get_earnings_dates(limit=limit)
                if ed is not None and not ed.empty:
                    all_dates.append(ed)
            except Exception:
                continue

        if not all_dates:
            print(f"  ⚠ {ticker}: 无法获取earnings_dates，尝试备用方法...")
            return get_earnings_from_calendar(stock, ticker)

        # 合并去重
        combined = pd.concat(all_dates)
        combined = combined[~combined.index.duplicated(keep='first')]

        results = []
        for date_idx in combined.index:
            ed = pd.Timestamp(date_idx)
            if ed.tzinfo:
                ed = ed.tz_localize(None)
            if START_DATE <= ed.to_pydatetime() <= END_DATE:
                results.append({
                    "ticker": ticker,
                    "company": MAG7[ticker],
                    "earnings_date": ed.strftime("%Y-%m-%d"),
                })

        return results

    except Exception as e:
        print(f"  ⚠ {ticker}: earnings_dates 获取失败 ({e})，尝试备用...")
        return get_earnings_from_calendar(stock, ticker)


def get_earnings_from_calendar(stock, ticker: str) -> list[dict]:
    """备用方法：从earnings_history或quarterly financials推断"""
    results = []
    try:
        # 用quarterly income statement的日期作为近似
        qf = stock.quarterly_income_stmt
        if qf is not None and not qf.empty:
            for col in qf.columns:
                ed = pd.Timestamp(col)
                if ed.tzinfo:
                    ed = ed.tz_localize(None)
                # 财报通常在季度结束后3-5周发布，加上35天作为估计
                approx_ed = ed + timedelta(days=35)
                if START_DATE <= approx_ed <= END_DATE:
                    results.append({
                        "ticker": ticker,
                        "company": MAG7[ticker],
                        "earnings_date": approx_ed.strftime("%Y-%m-%d"),
                        "note": "estimated from fiscal quarter end",
                    })
    except Exception as e:
        print(f"  ✗ {ticker}: 备用方法也失败 ({e})")

    return results


def compute_windows(earnings_events: list[dict]) -> pd.DataFrame:
    """为每个财报事件计算情绪窗口和静默期窗口"""
    records = []

    for event in earnings_events:
        ed = datetime.strptime(event["earnings_date"], "%Y-%m-%d")
        ticker = event["ticker"]
        company = event["company"]

        # ── 情绪窗口: [ED-7, ED-1] ──
        for offset in range(1, SENTIMENT_WINDOW + 1):
            day = ed - timedelta(days=offset)
            records.append({
                "ticker": ticker,
                "company": company,
                "collect_date": day.strftime("%Y-%m-%d"),
                "window_type": "sentiment",
                "earnings_date": event["earnings_date"],
                "offset_from_ed": -offset,
            })

        # ── 静默期窗口: [ED-37, ED-30] ──
        for offset in range(QUIET_END, QUIET_START + 1):
            day = ed - timedelta(days=offset)
            records.append({
                "ticker": ticker,
                "company": company,
                "collect_date": day.strftime("%Y-%m-%d"),
                "window_type": "quiet",
                "earnings_date": event["earnings_date"],
                "offset_from_ed": -offset,
            })

    return pd.DataFrame(records)


def build_spillover_schedule(self_schedule: pd.DataFrame,
                             earnings_events: list[dict]) -> pd.DataFrame:
    """
    对每个财报事件，还需要采集其余6家公司同期的新闻。
    只在情绪窗口做溢出采集（静默期的溢出采集也加上）。
    """
    records = []
    # 按事件分组
    events_by_key = {}
    for evt in earnings_events:
        key = (evt["ticker"], evt["earnings_date"])
        events_by_key[key] = evt

    for (ticker, ed_str), evt in events_by_key.items():
        ed = datetime.strptime(ed_str, "%Y-%m-%d")
        other_tickers = [t for t in MAG7 if t != ticker]

        for window_type, start_offset, end_offset in [
            ("sentiment_spillover", 1, SENTIMENT_WINDOW),
            ("quiet_spillover", QUIET_END, QUIET_START),
        ]:
            for offset in range(start_offset, end_offset + 1):
                day = ed - timedelta(days=offset)
                for other in other_tickers:
                    records.append({
                        "ticker": other,
                        "company": MAG7[other],
                        "collect_date": day.strftime("%Y-%m-%d"),
                        "window_type": window_type,
                        "earnings_date": ed_str,
                        "source_event_ticker": ticker,
                        "offset_from_ed": -offset,
                    })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("M7 财报日期采集 & 新闻采集计划生成")
    print("=" * 60)

    # ── Step 1: 获取所有财报日期 ──
    all_events = []
    for ticker in MAG7:
        print(f"\n📡 获取 {ticker} ({MAG7[ticker]}) 财报日期...")
        events = get_earnings_dates(ticker)
        print(f"  ✓ 找到 {len(events)} 个财报事件")
        for e in events:
            print(f"    {e['earnings_date']}", end="")
            if "note" in e:
                print(f" ({e['note']})", end="")
            print()
        all_events.extend(events)

    # 保存财报日期
    events_df = pd.DataFrame(all_events)
    events_path = os.path.join(OUTPUT_DIR, "earnings", "mag7_earnings_dates.csv")
    events_df.to_csv(events_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"总计: {len(all_events)} 个财报事件")
    print(f"已保存: {events_path}")

    # ── Step 2: 计算自身新闻采集窗口 ──
    print(f"\n计算新闻采集窗口...")
    self_schedule = compute_windows(all_events)
    print(f"  自身采集: {len(self_schedule)} 条 (公司,日期) 记录")

    # ── Step 3: 计算溢出采集窗口 ──
    spillover_schedule = build_spillover_schedule(self_schedule, all_events)
    print(f"  溢出采集: {len(spillover_schedule)} 条 (公司,日期) 记录")

    # ── Step 4: 合并 & 去重 ──
    full_schedule = pd.concat([self_schedule, spillover_schedule], ignore_index=True)
    print(f"  合并总计: {len(full_schedule)} 条记录")

    # 按 (ticker, collect_date) 去重，保留所有window_type信息
    dedup = full_schedule.drop_duplicates(subset=["ticker", "collect_date"])
    print(f"  去重后独立 (公司,日期): {len(dedup)} 条")

    # ── Step 5: 保存结果 ──
    schedule_path = os.path.join(OUTPUT_DIR, "news_schedule", "full_news_schedule.csv")
    full_schedule.to_csv(schedule_path, index=False)

    dedup_path = os.path.join(OUTPUT_DIR, "news_schedule", "unique_fetch_list.csv")
    dedup_sorted = dedup.sort_values(["ticker", "collect_date"])
    dedup_sorted[["ticker", "collect_date"]].to_csv(dedup_path, index=False)

    # ── Step 6: 统计摘要 ──
    print(f"\n{'=' * 60}")
    print("采集计划摘要")
    print(f"{'=' * 60}")

    print(f"\n财报事件数:")
    for ticker in MAG7:
        count = events_df[events_df["ticker"] == ticker].shape[0]
        print(f"  {ticker:6s} ({MAG7[ticker]:12s}): {count} 个季度")

    print(f"\n窗口类型分布:")
    for wt in full_schedule["window_type"].unique():
        count = full_schedule[full_schedule["window_type"] == wt].shape[0]
        print(f"  {wt:25s}: {count} 条")

    print(f"\n每家公司需采集的独立日期数:")
    for ticker in MAG7:
        count = dedup[dedup["ticker"] == ticker].shape[0]
        print(f"  {ticker:6s}: {count} 天")

    print(f"\n日期范围: {dedup['collect_date'].min()} ~ {dedup['collect_date'].max()}")
    print(f"\n总独立 (公司,日期) 查询数: {len(dedup)}")
    print(f"\n已保存完整计划: {schedule_path}")
    print(f"已保存去重列表: {dedup_path}")


if __name__ == "__main__":
    main()
