"""
Data loading and feature engineering for the AI Hype Decoded dashboard.
Reconstructs all model features and walk-forward predictions from raw CSV data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
MIN_TRAIN = 40
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


# ── Data Loading ──────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "prices" / "daily_prices.csv", parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_earnings() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "earnings" / "mag7_earnings.csv", parse_dates=["earnings_date"])
    return df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)


def load_daily_sentiment() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sentiment" / "daily_sentiment.csv", parse_dates=["date"])
    return df


def load_window_articles() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "news" / "window_articles.csv",
                      parse_dates=["earnings_date", "article_date"])
    return df


def load_m3_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "spillover" / "m3_features.csv",
                      parse_dates=["earnings_date"])
    return df


def load_network_snapshots() -> list:
    with open(DATA_DIR / "spillover" / "network_snapshots.json") as f:
        return json.load(f)


# ── Feature Engineering ───────────────────────────────────────────────────

def build_base_df(earnings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Build base DataFrame with target variable (5-day post-earnings return)."""
    rows = []
    for _, ev in earnings.iterrows():
        tk = ev["ticker"]
        ed = ev["earnings_date"]
        tk_prices = prices[prices["ticker"] == tk].sort_values("date")

        # p0: close on or before earnings date
        pre = tk_prices[tk_prices["date"] <= ed]
        if pre.empty:
            continue
        p0 = pre.iloc[-1]["close"]

        # p5: close 5 trading days after ED
        post = tk_prices[tk_prices["date"] > ed]
        if len(post) < 5:
            continue
        p5 = post.iloc[4]["close"]

        ret_5d = (p5 - p0) / p0
        rows.append({
            "ticker": tk,
            "company": ev["company"],
            "earnings_date": ed,
            "eps_estimate": ev["eps_estimate"],
            "eps_actual": ev["eps_actual"],
            "surprise_pct": ev["surprise_pct"],
            "ret_5d": ret_5d,
            "target": int(ret_5d > 0),
        })
    return pd.DataFrame(rows).sort_values("earnings_date").reset_index(drop=True)


def add_sentiment_features(df: pd.DataFrame, articles: pd.DataFrame,
                           daily_sent: pd.DataFrame) -> pd.DataFrame:
    """Add M2 sentiment features: sent_mean, sent_trend, sent_delta, news_volume."""
    sent_rows = []
    for _, ev in df.iterrows():
        tk, ed = ev["ticker"], ev["earnings_date"]

        # Window articles [ED-7, ED-1]
        window = articles[(articles["ticker"] == tk) & (articles["earnings_date"] == ed)]

        if len(window) < 3:
            sent_rows.append({"ticker": tk, "earnings_date": ed,
                              "sent_mean": np.nan, "sent_trend": np.nan,
                              "sent_delta": np.nan, "news_volume": len(window)})
            continue

        sent_mean = window["polarity"].mean()
        news_volume = len(window)

        # Trend: late half mean - early half mean
        days = window.groupby("article_date")["polarity"].mean().sort_index()
        mid = len(days) // 2
        if mid > 0 and mid < len(days):
            sent_trend = days.iloc[mid:].mean() - days.iloc[:mid].mean()
        else:
            sent_trend = 0.0

        # Delta vs quiet period [ED-37, ED-30]
        quiet_start = ed - pd.Timedelta(days=37)
        quiet_end = ed - pd.Timedelta(days=30)
        quiet = daily_sent[(daily_sent["ticker"] == tk) &
                           (daily_sent["date"] >= quiet_start) &
                           (daily_sent["date"] <= quiet_end)]
        if len(quiet) >= 3:
            sent_delta = sent_mean - quiet["sentiment"].mean()
        else:
            sent_delta = np.nan

        sent_rows.append({"ticker": tk, "earnings_date": ed,
                          "sent_mean": sent_mean, "sent_trend": sent_trend,
                          "sent_delta": sent_delta, "news_volume": news_volume})

    sent_df = pd.DataFrame(sent_rows)
    return df.merge(sent_df, on=["ticker", "earnings_date"], how="left")


def add_spillover_features(df: pd.DataFrame, m3_feat: pd.DataFrame) -> pd.DataFrame:
    """Merge M3 spillover features."""
    m3_cols = ["spillover_weighted_sent", "net_transmitter",
               "system_connectedness", "spillover_neg_asym"]
    return df.merge(m3_feat[["ticker", "earnings_date"] + m3_cols],
                    on=["ticker", "earnings_date"], how="left")


# ── Walk-Forward Model Predictions ────────────────────────────────────────

M2_FEATURES = ["surprise_pct", "sent_mean", "sent_trend", "sent_delta", "news_volume"]
M3_FEATURES = M2_FEATURES + ["spillover_weighted_sent", "net_transmitter",
                              "system_connectedness", "spillover_neg_asym"]


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run walk-forward predictions for all 5 models.
    Returns df with added columns: m0_pred, m1_pred, m2_pred, m3_pred, xgb_pred,
    m0_prob, m1_prob, m2_prob, m3_prob, xgb_prob, and is_test flag.
    """
    n = len(df)
    y = df["target"].values

    # Initialize prediction columns
    for model in ["m0", "m1", "m2", "m3", "xgb"]:
        df[f"{model}_pred"] = np.nan
        df[f"{model}_prob"] = np.nan
    df["is_test"] = False

    # Check which rows have full features
    has_m2 = df[M2_FEATURES].notna().all(axis=1).values
    has_m3 = df[M3_FEATURES].notna().all(axis=1).values

    for i in range(MIN_TRAIN, n):
        train_idx = np.arange(0, i)
        y_train = y[train_idx]
        df.loc[df.index[i], "is_test"] = True

        # M0: naive rule
        df.loc[df.index[i], "m0_pred"] = int(df.iloc[i]["surprise_pct"] > 0)
        df.loc[df.index[i], "m0_prob"] = float(df.iloc[i]["surprise_pct"] > 0)

        # M1: LogReg on surprise_pct
        X_m1 = df.iloc[train_idx][["surprise_pct"]].values
        lr1 = LogisticRegression(random_state=42, max_iter=1000)
        lr1.fit(X_m1, y_train)
        x_test_m1 = df.iloc[[i]][["surprise_pct"]].values
        df.loc[df.index[i], "m1_pred"] = int(lr1.predict(x_test_m1)[0])
        df.loc[df.index[i], "m1_prob"] = float(lr1.predict_proba(x_test_m1)[0, 1])

        # M2: LogReg on surprise + sentiment (only if features available)
        if has_m2[i]:
            train_m2 = train_idx[has_m2[train_idx]]
            if len(train_m2) >= 20:
                X_train_m2 = df.iloc[train_m2][M2_FEATURES].values
                y_train_m2 = y[train_m2]
                scaler2 = StandardScaler()
                X_train_m2_s = scaler2.fit_transform(X_train_m2)
                lr2 = LogisticRegression(random_state=42, max_iter=1000)
                lr2.fit(X_train_m2_s, y_train_m2)
                x_test_m2 = scaler2.transform(df.iloc[[i]][M2_FEATURES].values)
                df.loc[df.index[i], "m2_pred"] = int(lr2.predict(x_test_m2)[0])
                df.loc[df.index[i], "m2_prob"] = float(lr2.predict_proba(x_test_m2)[0, 1])

        # M3 + XGB: LogReg and XGBoost on 9 features (only if spillover available)
        if has_m3[i]:
            train_m3 = train_idx[has_m3[train_idx]]
            if len(train_m3) >= 20:
                X_train_m3 = df.iloc[train_m3][M3_FEATURES].values
                y_train_m3 = y[train_m3]
                scaler3 = StandardScaler()
                X_train_m3_s = scaler3.fit_transform(X_train_m3)
                x_test_m3 = scaler3.transform(df.iloc[[i]][M3_FEATURES].values)

                # M3 LogReg
                lr3 = LogisticRegression(random_state=42, max_iter=1000)
                lr3.fit(X_train_m3_s, y_train_m3)
                df.loc[df.index[i], "m3_pred"] = int(lr3.predict(x_test_m3)[0])
                df.loc[df.index[i], "m3_prob"] = float(lr3.predict_proba(x_test_m3)[0, 1])

                # XGBoost
                xgb = XGBClassifier(
                    n_estimators=50, max_depth=2, learning_rate=0.05,
                    min_child_weight=5, subsample=0.8, random_state=42,
                    eval_metric="logloss", verbosity=0
                )
                xgb.fit(X_train_m3_s, y_train_m3)
                df.loc[df.index[i], "xgb_pred"] = int(xgb.predict(x_test_m3)[0])
                df.loc[df.index[i], "xgb_prob"] = float(xgb.predict_proba(x_test_m3)[0, 1])

    return df


def get_model_coefficients(df: pd.DataFrame) -> dict:
    """Train final models on all available data to get coefficients for display."""
    has_m3 = df[M3_FEATURES].notna().all(axis=1)
    df_m3 = df[has_m3].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_m3[M3_FEATURES].values)
    y = df_m3["target"].values

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    return dict(zip(M3_FEATURES, lr.coef_[0]))


# ── Master Loader ─────────────────────────────────────────────────────────

def load_all_data():
    """Load and process all data. Returns (prices, events_df, network_snapshots)."""
    prices = load_prices()
    earnings = load_earnings()
    daily_sent = load_daily_sentiment()
    articles = load_window_articles()
    m3_feat = load_m3_features()
    snapshots = load_network_snapshots()

    df = build_base_df(earnings, prices)
    df = add_sentiment_features(df, articles, daily_sent)
    df = add_spillover_features(df, m3_feat)
    df = run_walk_forward(df)

    return prices, df, snapshots
