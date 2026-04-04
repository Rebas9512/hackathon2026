# AI Hype Decoded: Multi-Source Sentiment Spillover and Stock Price Prediction

> 2026 FinHack Challenge - Case 4  
> UTD JSOM Finance Lab

## Project Scope

Build a sentiment-driven prediction system that tests whether AI-related text signals can improve **post-earnings stock price prediction** for the Magnificent 7 companies, with a focus on modeling **cross-company sentiment spillover networks**.

### Target Companies (Magnificent 7)

AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet), AMZN (Amazon), NVDA (NVIDIA), META (Meta), TSLA (Tesla)

### Prediction Task

- **Target**: 5-trading-day cumulative return direction after earnings (binary: up/down)
- **Time Range**: 2022 Q1 - 2025 Q1 (91 earnings events, 7 companies x 13 quarters)

---

## Three-Model Progressive Comparison

The core deliverable is a **progressive model comparison** that isolates the incremental value of each signal layer:

| Model | Features | Purpose |
|-------|----------|---------|
| **M1: Baseline** | Earnings + market features only | "How much can we predict without sentiment?" |
| **M2: + Direct Sentiment** | + company-specific 7-day sentiment | "Does sentiment about this company help?" |
| **M3: + Spillover Network** | + dual-layer DY network cross-company signals | "Does cross-company spillover help?" |

---

## Pipeline Design

### Phase 1: Data Collection (Complete)

| Data | Source | Format | Status |
|------|--------|--------|--------|
| Daily stock prices (OHLCV) | yfinance | M7+SPY, 962 trading days (2021-06 ~ 2025-03) | ✅ |
| Earnings dates + EPS surprise | yfinance | 91 events, 100% EPS coverage | ✅ |
| Daily aggregated sentiment | EODHD `/api/sentiments` | 7 tickers × ~1,300 days continuous (2021-06 ~ 2025-03) | ✅ |
| Pre-earnings window news | EODHD `/api/news` | 13,588 articles in [ED-7, ED-1] with per-article scores | ✅ |

### Phase 2: Time Window System

Each earnings event `(Company, Earnings_Date)` defines windows:

```
Timeline:
──[ED-150]───────[ED-37, ED-30]────[ED-7, ED-1]──[ED]──[ED+1, ED+5]───→

     |                 |                  |          |        |
  DY Network      Quiet Period       Sentiment     Skip    Target
  Window          (baseline)         Window        Day     Variable
  [ED-150, ED-8]  [ED-37, ED-30]    [ED-7, ED-1]         5-day return
```

- **Sentiment Window** `[ED-7, ED-1]`: 7 days of pre-earnings news sentiment (M2/M3 features)
- **Quiet Period** `[ED-37, ED-30]`: 8 days of baseline sentiment from daily aggregated data (for delta/anomaly detection)
- **DY Network Window** `[ED-150, ED-8]`: ~142 days for dual-layer VAR-GFEVD spillover network (M3 features)
- **ED Day**: Completely excluded (cannot reliably separate pre/post-earnings news)
- **Target** `[ED+1, ED+5]`: 5-trading-day cumulative return, label = 1 if positive

### Phase 3: Sentiment Data & Methodology

#### EODHD Sentiment Scoring

EODHD provides sentiment analysis on financial news articles using a VADER-consistent NLP model. Scores are computed on full article body text.

**Per-article scores** (from `/api/news`, used in M2 window features):

| Field | Range | Description |
|-------|-------|-------------|
| `polarity` | [-1.0, +1.0] | Compound sentiment score (analogous to VADER compound) |
| `neg` | [0.0, 1.0] | Proportion of text classified as negative |
| `neu` | [0.0, 1.0] | Proportion of text classified as neutral |
| `pos` | [0.0, 1.0] | Proportion of text classified as positive |

Note: `neg + neu + pos = 1.0` (proportional decomposition). `polarity` is an independent compound score, not a simple function of the three proportions.

**Daily aggregated scores** (from `/api/sentiments`, used in DY network and quiet period):

| Field | Range | Description |
|-------|-------|-------------|
| `normalized` | [-1.0, +1.0] | Arithmetic mean of all article `polarity` scores for that ticker on that date |
| `count` | integer | Number of articles analyzed |

#### Two-Level Sentiment Data Strategy

| Level | Data Source | Used For |
|-------|------------|----------|
| **Article-level** (`window_articles.csv`) | `/api/news` for [ED-7, ED-1] windows | M2 features: precise sent_mean, sent_trend, news_volume from per-article polarity |
| **Daily-aggregate** (`daily_sentiment.csv`) | `/api/sentiments` continuous series | DY sentiment network input, quiet period baseline, system-level analysis |

### Phase 4: Feature Engineering

**M1 - Baseline Features (earnings + market)**
- `eps_surprise_pct`: actual vs consensus EPS
- `ret_5d`, `ret_20d`: pre-earnings momentum
- `volatility_20d`: realized volatility
- `spy_ret_5d`: market context
- `relative_strength`: stock vs SPY

**M2 - Direct Sentiment Features (company-specific, 7-day window)**

Computed from **article-level** data (`window_articles.csv`):
- `sent_mean`: average polarity over [ED-7, ED-1] articles
- `sent_trend`: sentiment slope (later 3 days mean - earlier 4 days mean)
- `news_volume`: total number of articles in sentiment window

Computed using **daily-aggregate** data (`daily_sentiment.csv`) for quiet period:
- `sent_delta`: sent_mean - quiet_period_mean (anomaly signal vs [ED-37, ED-30] baseline)

**M3 - Spillover Network Features (cross-company, dual-layer DY framework)**
- `spillover_weighted_sent`: Σ_j W[i,j] × sent_mean_j (other companies' sentiment weighted by composite network)
- `net_transmitter`: out_degree_i - in_degree_i (is this company a sender or receiver of shocks?)
- `system_connectedness`: total off-diagonal sum / N (how tightly coupled is the M7 system?)
- `spillover_neg_asym`: weighted sum of spillover from companies with negative sentiment (asymmetric effect, per Nyakurukwa & Seetharam 2025)

### Phase 5: Dual-Layer Spillover Network — Diebold-Yilmaz Framework (M3 Core)

The key innovation: with continuous daily sentiment data for all 7 companies, we can now build **dual-layer** dynamic connectedness networks — one for returns and one for sentiment — following the Diebold-Yilmaz (2014) framework.

#### Layer 1: Return Connectedness

For each earnings event, using [ED-150, ED-8] **daily returns** of 7 M7 stocks (~142 trading days):

```
VAR(p) on M7 daily returns → Generalized FEVD (Pesaran-Shin 1998)
→ d_ij_return = fraction of i's return forecast error variance explained by j's shocks
→ Produces weighted, directed return spillover network
```

#### Layer 2: Sentiment Connectedness

For each earnings event, using [ED-150, ED-8] **daily aggregated sentiment** of 7 M7 stocks (~142 calendar days):

```
VAR(p) on M7 daily sentiment → Generalized FEVD
→ d_ij_sentiment = "whose sentiment change predicts whose?"
→ Captures sentiment contagion structure (e.g., negative AI news spreading from NVDA to others)
```

#### Composite Spillover Weight

```
W_ij = α × d_ij_return + β × d_ij_sentiment
```

Where α, β are tuned or set to equal weights (0.5, 0.5).

#### Implementation Details

```
Step 1: Estimate VAR(p) — statsmodels, BIC order selection (expect p=1 or 2)
Step 2: Compute Generalized FEVD (H=10) — Pesaran-Shin (1998), custom implementation
Step 3: Normalize → d_ij / row_sum → adjacency matrix of weighted directed network
Step 4: Repeat for both return and sentiment layers
Step 5: Combine into composite W_ij, extract network features
```

**Fallback**: If GFEVD proves unstable, fall back to **rolling correlation matrices** as network weights. This preserves the spillover story with simpler implementation.

**Key outputs for visualization:**
- 7×7 spillover heatmap per quarter (return layer vs sentiment layer)
- Dynamic network graph (node size = net transmitter score, edge width = W_ij)
- Time-varying total system connectedness curve (with AI event annotations)
- NVDA's evolving role: receiver (2022) → transmitter (2024)
- Asymmetric analysis: negative vs positive sentiment transmission strength

### Phase 6: Modeling & Evaluation

**Algorithms:**
- Primary: Logistic Regression (91 samples → simple models preferred)
- Secondary: XGBoost with strict regularization (robustness check)

**Validation: Time-ordered expanding window walk-forward**
- Never shuffle samples
- Train on first T events → predict event T+1 → expand window
- Start predicting after ~40 events (minimum training set)

**Metrics:**
- Accuracy, AUC-ROC, F1
- Simulated trading returns (long if predicted up, short if predicted down)
- Statistical test for M1 vs M2 vs M3 differences (McNemar test)

### Phase 7: Visualization / Dashboard

Tech: Streamlit + Plotly

1. **Sentiment timeline** vs stock price overlay (per company)
2. **Pre-earnings sentiment vs quiet period** comparison
3. **Dynamic M7 network graph** (interactive, switchable by quarter, return vs sentiment layer)
4. **Spillover heatmap** by quarter (showing NVDA's rise to center)
5. **Time-varying total connectedness** curve (with AI event annotations)
6. **M1 → M2 → M3 model comparison** (AUC, accuracy, feature importance)
7. **Single event drill-down** (e.g., NVDA 2024 Q2 earnings)
8. **Simulated trading returns** curve

---

## Data Leakage Controls

- Sentiment data strictly cut off at ED-1
- ED day news completely excluded
- DY network window [ED-150, ED-8] does not overlap with sentiment window [ED-7, ED-1]
- Quiet period [ED-37, ED-30] far from both sentiment window and earnings date
- Walk-forward validation, never shuffle samples
- All features use only pre-ED data

---

## Implementation Risk & Fallback

| Component | Risk | Fallback |
|-----------|------|----------|
| GFEVD implementation | High - custom Pesaran-Shin needed, VAR stability on 7-dim | Rolling correlation matrix as network weights |
| Dual-layer DY (sentiment layer) | Medium - weekend gaps in sentiment data, lower news count days | Filter to trading days only; or use return-layer-only network |
| 91 sample size → overfitting | Medium | Logistic Regression only, strict feature count limits |
| M2/M3 improvement not significant | Medium | Frame as "evidence that sentiment doesn't add value" — still a valid finding |
| META early data missing (2022 Q1-Q2) | Low - 2-3 events affected | Exclude from M2/M3, keep in M1 |

---

## Academic References

- Diebold, F.X. and K. Yilmaz (2014). "On the Network Topology of Variance Decompositions: Measuring the Connectedness of Financial Firms." *Journal of Econometrics*, 182, 119-134.
- Nyakurukwa, K. and Y. Seetharam (2025). "Investor Sentiment Networks: Mapping Connectedness in DJIA Stocks." *Financial Innovation*, 11:4.
- Pesaran, M.H. and Y. Shin (1998). "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters*, 58, 17-29.
- Hutto, C.J. and E. Gilbert (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *ICWSM*.

---

## Project Structure

```
hackathon2026/
├── README.md
├── .env                              # API keys (not committed)
├── case/                             # Challenge case PDF
├── papers/                           # Reference papers (DY 2014, Nyakurukwa 2025)
├── data/
│   ├── hackathon.db                      # SQLite database (all tables, 3.2 MB, gitignored)
│   ├── earnings/
│   │   └── mag7_earnings.csv             # 91 events + EPS surprise
│   ├── prices/
│   │   └── daily_prices.csv              # M7+SPY daily OHLCV, 2021-06~2025-03
│   ├── sentiment/
│   │   ├── daily_sentiment.csv           # Daily aggregated sentiment, 2021-06~2025-03
│   │   └── extreme_events.csv           # Notable sentiment events for dashboard
│   └── news/
│       └── window_articles.csv           # 13,588 articles in [ED-7,ED-1] windows
├── scripts/
│   ├── 01_fetch_sentiment.py             # EODHD daily sentiment
│   ├── 02_fetch_prices.py               # yfinance prices + EPS
│   ├── 03_fetch_window_news.py          # EODHD window news articles
│   ├── build_db.py                      # Build SQLite from CSVs
│   ├── 04_features.py                   # Feature engineering (pending)
│   ├── 05_spillover_network.py          # DY framework / correlation fallback (pending)
│   ├── 06_model.py                      # Model training & evaluation (pending)
│   └── 07_dashboard.py                  # Streamlit dashboard (pending)
├── notebooks/                            # Analysis notebooks (pending)
└── app/                                 # Streamlit dashboard (pending)
```

> **Note**: Run `build_db.py` after any CSV changes to rebuild the SQLite database.

## Current Status

- [x] Project scope and pipeline design finalized
- [x] Three-model progressive comparison designed (M1 → M2 → M3)
- [x] Earnings data collected (91 events, 100% EPS surprise coverage)
- [x] Stock price data collected (M7 + SPY, 962 trading days, zero missing)
- [x] Daily sentiment collected (7 tickers × ~1,300 days continuous series)
- [x] Window news articles collected (13,588 articles with per-article polarity)
- [x] Dual-layer DY network approach designed (return + sentiment connectedness)
- [x] Sentiment methodology documented (EODHD VADER-consistent scoring)
- [ ] Feature engineering (M1/M2/M3 features)
- [ ] DY spillover network computation (or correlation fallback)
- [ ] Model training and walk-forward evaluation
- [ ] Dashboard / visualization
- [ ] Presentation
