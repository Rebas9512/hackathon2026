"""
04_build_spillover_network.py

Compute dual-layer Diebold-Yilmaz spillover networks (returns + sentiment)
for each earnings event, then extract M3 features.

Pipeline per event:
  1. Slice [ED-150, ED-8] window for returns and sentiment
  2. Fit VAR(p) on each layer (p chosen by BIC, floor=1)
  3. Compute Generalized FEVD (Pesaran-Shin 1998) → 7×7 matrix per layer
  4. Combine: W = α·D_return + β·D_sentiment (default α=β=0.5)
  5. Extract 4 M3 features for the target company

Outputs:
  - data/spillover/connectedness_matrices.pkl   (all 7×7 matrices per event)
  - data/spillover/m3_features.csv              (M3 features ready to merge)
"""

import os
import warnings
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
N = len(TICKERS)

# ── Hyperparameters ──────────────────────────────────────────────────────
WINDOW_DAYS = 150          # calendar days back from ED
GAP_DAYS = 8               # exclude [ED-7, ED] to avoid overlap with M2
FORECAST_HORIZON = 10      # H-step GFEVD
VAR_MAXLAGS = 5            # max lags for BIC selection
ALPHA = 0.5                # return layer weight
BETA = 0.5                 # sentiment layer weight
MIN_OBS = 80               # minimum trading days required in window


# ── Data Loading ─────────────────────────────────────────────────────────

def load_data():
    """Load and pivot returns + sentiment into date×ticker matrices."""
    # Returns
    prices = pd.read_csv(
        os.path.join(BASE_DIR, "data/prices/daily_prices.csv"),
        parse_dates=["date"],
    )
    prices = prices[prices["ticker"].isin(TICKERS)]
    prices = prices.sort_values(["ticker", "date"])
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    ret_pivot = prices.pivot(index="date", columns="ticker", values="ret")[TICKERS]

    # Sentiment
    sent = pd.read_csv(
        os.path.join(BASE_DIR, "data/sentiment/daily_sentiment.csv"),
        parse_dates=["date"],
    )
    sent_pivot = sent.pivot(index="date", columns="ticker", values="sentiment")[TICKERS]
    # Forward-fill gaps (max 5 days) then drop remaining NaN rows
    sent_pivot = sent_pivot.ffill(limit=5)

    # Earnings
    earnings = pd.read_csv(
        os.path.join(BASE_DIR, "data/earnings/mag7_earnings.csv"),
        parse_dates=["earnings_date"],
    )
    earnings = earnings.sort_values("earnings_date").reset_index(drop=True)

    return ret_pivot, sent_pivot, earnings


# ── Generalized FEVD (Pesaran-Shin 1998) ─────────────────────────────────

def generalized_fevd(var_result, H=10):
    """
    Compute the Generalized Forecast Error Variance Decomposition.

    Returns a normalized N×N matrix D where D[i][j] = fraction of variable i's
    H-step forecast error variance explained by shocks to variable j.
    Rows sum to 1.
    """
    ma = var_result.ma_rep(H)      # (H+1, N, N) MA coefficient matrices
    sigma = var_result.sigma_u     # (N, N) residual covariance
    n = sigma.shape[0]
    sigma_diag = np.diag(sigma)    # σ_jj for each variable

    theta = np.zeros((n, n))

    for i in range(n):
        # Denominator: total forecast error variance for variable i
        denom = 0.0
        for h in range(H):
            psi_h = ma[h]
            denom += psi_h[i] @ sigma @ psi_h[i]

        for j in range(n):
            # Numerator: contribution of shock j to variable i
            numer = 0.0
            for h in range(H):
                psi_h = ma[h]
                val = psi_h[i] @ sigma[:, j]
                numer += val ** 2
            theta[i, j] = numer / (sigma_diag[j] * denom)

    # Row-normalize so each row sums to 1
    D = theta / theta.sum(axis=1, keepdims=True)
    return D


# ── VAR + GFEVD for One Layer ────────────────────────────────────────────

def compute_connectedness(data_matrix):
    """
    Fit VAR and compute GFEVD on a T×N numpy array.
    Returns the N×N connectedness matrix, or None if estimation fails.
    """
    try:
        model = VAR(data_matrix)
        # Select lag by BIC, but ensure at least 1
        result = model.fit(maxlags=VAR_MAXLAGS, ic="bic")
        if result.k_ar == 0:
            result = model.fit(1)
        D = generalized_fevd(result, H=FORECAST_HORIZON)
        return D
    except Exception as e:
        print(f"    VAR failed: {e}")
        return None


# ── Feature Extraction ───────────────────────────────────────────────────

def extract_m3_features(W, ticker, sent_means, sent_medians):
    """
    Extract 4 M3 features from composite connectedness matrix W.

    Args:
        W:              N×N connectedness matrix
        ticker:         target company ticker
        sent_means:     dict {ticker: sent_mean in [ED-7, ED-1]} for all 7 companies
        sent_medians:   dict {ticker: historical median sentiment} for baseline

    Returns:
        dict with 4 feature values
    """
    idx = TICKERS.index(ticker)

    # In-degree: how much others affect me (row sum minus diagonal)
    in_degree = W[idx].sum() - W[idx, idx]

    # Out-degree: how much I affect others (column sum minus diagonal)
    out_degree = W[:, idx].sum() - W[idx, idx]

    # Feature 1: spillover_weighted_sent
    # Weighted average of other companies' sentiment, weighted by spillover to me
    weighted_sent = 0.0
    weight_sum = 0.0
    for j, t in enumerate(TICKERS):
        if j == idx:
            continue
        if t in sent_means:
            weighted_sent += W[idx, j] * sent_means[t]
            weight_sum += W[idx, j]
    spillover_weighted_sent = weighted_sent / weight_sum if weight_sum > 0 else np.nan

    # Feature 2: net_transmitter
    net_transmitter = out_degree - in_degree

    # Feature 3: system_connectedness (same for all companies in this event)
    system_connectedness = (W.sum() - np.trace(W)) / N

    # Feature 4: spillover_neg_asym
    # Weighted sum of BELOW-MEDIAN sentiment from others.
    # Raw sentiment is almost always > 0 for Mag7, so we measure "relatively
    # bearish" as (median - current) clipped at 0. Higher = more bearish
    # spillover pressure from companies that influence me.
    neg_asym = 0.0
    for j, t in enumerate(TICKERS):
        if j == idx:
            continue
        if t in sent_means and t in sent_medians:
            bearish_gap = max(0.0, sent_medians[t] - sent_means[t])
            neg_asym += W[idx, j] * bearish_gap
    spillover_neg_asym = neg_asym

    return {
        "spillover_weighted_sent": spillover_weighted_sent,
        "net_transmitter": net_transmitter,
        "system_connectedness": system_connectedness,
        "spillover_neg_asym": spillover_neg_asym,
    }


# ── Sentiment Means for [ED-7, ED-1] ────────────────────────────────────

def get_sent_means(sent_pivot, ed):
    """Get mean sentiment in [ED-7, ED-1] for each ticker."""
    start = ed - pd.Timedelta(days=7)
    end = ed - pd.Timedelta(days=1)
    window = sent_pivot.loc[start:end]
    means = {}
    for t in TICKERS:
        vals = window[t].dropna()
        if len(vals) >= 2:
            means[t] = vals.mean()
    return means


# ── Main Pipeline ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("M3 Spillover Network — Diebold-Yilmaz Dual-Layer Pipeline")
    print("=" * 60)

    ret_pivot, sent_pivot, earnings = load_data()
    print(f"Returns: {ret_pivot.shape[0]} trading days")
    print(f"Sentiment: {sent_pivot.shape[0]} days")
    print(f"Earnings events: {len(earnings)}")

    # Historical median sentiment per ticker (baseline for neg_asym)
    sent_medians = {t: sent_pivot[t].median() for t in TICKERS}
    print(f"Sentiment medians: { {t: f'{v:.3f}' for t, v in sent_medians.items()} }")

    results = []
    matrices = {}
    skipped = 0

    for i, row in earnings.iterrows():
        ticker = row["ticker"]
        ed = row["earnings_date"]
        tag = f"{ticker} {ed.strftime('%Y-%m-%d')}"

        # ── Slice window [ED-150, ED-8] ──
        win_start = ed - pd.Timedelta(days=WINDOW_DAYS)
        win_end = ed - pd.Timedelta(days=GAP_DAYS)

        ret_window = ret_pivot.loc[win_start:win_end].dropna()
        sent_window = sent_pivot.loc[win_start:win_end].dropna()

        if len(ret_window) < MIN_OBS:
            print(f"  [{tag}] SKIP — only {len(ret_window)} return obs (need {MIN_OBS})")
            skipped += 1
            continue

        if len(sent_window) < MIN_OBS:
            print(f"  [{tag}] SKIP — only {len(sent_window)} sentiment obs (need {MIN_OBS})")
            skipped += 1
            continue

        # ── Compute connectedness for each layer ──
        D_ret = compute_connectedness(ret_window.values)
        D_sent = compute_connectedness(sent_window.values)

        if D_ret is None or D_sent is None:
            print(f"  [{tag}] SKIP — VAR estimation failed")
            skipped += 1
            continue

        # ── Composite network ──
        W = ALPHA * D_ret + BETA * D_sent

        # ── Extract features ──
        sent_means = get_sent_means(sent_pivot, ed)
        features = extract_m3_features(W, ticker, sent_means, sent_medians)

        features["ticker"] = ticker
        features["earnings_date"] = ed.strftime("%Y-%m-%d")
        features["ret_obs"] = len(ret_window)
        features["sent_obs"] = len(sent_window)
        features["var_lag_ret"] = "ok"
        features["var_lag_sent"] = "ok"
        results.append(features)

        matrices[tag] = {
            "D_return": D_ret,
            "D_sentiment": D_sent,
            "W_composite": W,
        }

        # Log a compact summary
        sys_c = features["system_connectedness"]
        net_t = features["net_transmitter"]
        sw_s = features["spillover_weighted_sent"]
        print(f"  [{tag}] sys_conn={sys_c:.3f}  net_trans={net_t:+.3f}  spill_sent={sw_s:.3f}")

    # ── Save outputs ──
    out_dir = os.path.join(BASE_DIR, "data", "spillover")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "m3_features.csv")
    df.to_csv(csv_path, index=False)

    pkl_path = os.path.join(out_dir, "connectedness_matrices.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(matrices, f)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Done. {len(results)} events computed, {skipped} skipped.")
    print(f"  Features → {csv_path}")
    print(f"  Matrices → {pkl_path}")

    if len(df) > 0:
        print(f"\nFeature summary:")
        for col in ["spillover_weighted_sent", "net_transmitter",
                     "system_connectedness", "spillover_neg_asym"]:
            print(f"  {col:30s}: mean={df[col].mean():.4f}  std={df[col].std():.4f}"
                  f"  min={df[col].min():.4f}  max={df[col].max():.4f}")


if __name__ == "__main__":
    main()
