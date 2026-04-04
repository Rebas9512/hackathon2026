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

## Pipeline Design

### Phase 1: Data Collection

| Data | Source | Cost |
|------|--------|------|
| Daily stock prices (OHLCV) | yfinance | Free |
| Earnings dates + EPS surprise | yfinance | Free |
| Financial statements | yfinance | Free |
| Market benchmark (SPY) | yfinance | Free |
| News articles + sentiment scores | EODHD API | $19.99/mo |

### Phase 2: Time Window System

Each earnings event `(Company, Earnings_Date)` defines four windows:

```
Timeline:
─────[ED-90]──────────[ED-37, ED-30]────[ED-7, ED-1]──[ED]──[ED+1, ED+5]───→

      |                    |                   |          |        |
 Correlation          Quiet Period        Sentiment     Skip    Target
 Matrix Window        (baseline)          Window        Day     Variable
 [ED-90, ED-8]       [ED-37, ED-30]      [ED-7, ED-1]         5-day return
```

- **Sentiment Window** `[ED-7, ED-1]`: 7 days of pre-earnings news sentiment (primary features)
- **Quiet Period** `[ED-37, ED-30]`: 8 days of baseline sentiment (for delta/anomaly detection)
- **Correlation Window** `[ED-90, ED-8]`: 90 days of returns for dynamic relationship modeling
- **ED Day**: Completely excluded (cannot reliably separate pre/post-earnings news)
- **Target** `[ED+1, ED+5]`: 5-trading-day cumulative return, label = 1 if positive

### Phase 3: Sentiment Processing

- Primary: EODHD built-in polarity scores (-1 to +1)
- Optional cross-validation: FinBERT on headlines

### Phase 4: Feature Engineering

**Layer 0 - Baseline (no sentiment)**
- `ret_5d`, `ret_20d`, `volatility_20d`, `spy_ret_5d`, `relative_strength`

**Layer 1 - Direct Sentiment (company-specific)**
- `sent_mean`, `sent_std`, `sent_trend`, `news_volume`
- `sent_delta` = sentiment window mean - quiet period mean (anomaly signal)
- `volume_delta` = news volume anomaly

**Layer 2 - Spillover (cross-company, core differentiator)**
- `spillover_weighted`, `spillover_max`, `spillover_dispersion`
- `sector_sentiment`, `centrality_score`, `spillover_delta`
- `net_transmitter_score`, `system_connectedness`
- `spillover_asymmetry` (negative vs positive sentiment transmission)

### Phase 5: Spillover Network Modeling (Core Innovation)

The project's key differentiator is a **multi-layer dynamic connectedness network** based on the Diebold-Yilmaz (2014) framework, applied to both returns and sentiment:

**Layer 1: Return Connectedness (Diebold-Yilmaz)**
```
VAR(p) on M7 daily returns
  -> Generalized Forecast Error Variance Decomposition (GFEVD)
  -> d_ij = "fraction of i's forecast error variance explained by shocks from j"
  -> Produces weighted, directed spillover network
  -> Rolling window: recomputed for each earnings event
```

**Layer 2: Sentiment Connectedness**
```
VAR(p) on M7 daily sentiment scores
  -> GFEVD on sentiment time series
  -> Captures "whose sentiment change predicts whose?"
  -> Asymmetric decomposition: sent+ vs sent- (negative sentiment spreads faster)
```

**Composite spillover weight:**
```
W_ij = alpha * d_ij_return + beta * d_ij_sentiment
```

**Key outputs:**
- Net transmitter vs receiver classification (e.g., NVDA transitions from receiver to top transmitter 2022->2024)
- Time-varying total system connectedness (spikes during major AI events)
- Directional spillover: "NVDA -> MSFT spillover strength = X%"
- Asymmetric analysis: negative AI news spreads wider than positive

### Phase 6: Modeling & Evaluation

| Model | Features | Purpose |
|-------|----------|---------|
| M0: Baseline | Market features only | "How much can we predict without sentiment?" |
| M1: + Direct Sentiment | + company sentiment | Direct sentiment value |
| M2: + Spillover | + cross-company spillover | Spillover incremental value |
| M3: + Delta | + quiet period anomaly | Anomaly signal value |

- Algorithms: Logistic Regression -> XGBoost (small sample, prefer simple models)
- Validation: Time-ordered expanding window walk-forward (never shuffle)
- Metrics: Accuracy, AUC-ROC, F1 + simulated trading returns

### Phase 7: Visualization / Dashboard

1. Sentiment timeline vs stock price overlay (per company)
2. Pre-earnings sentiment vs quiet period comparison
3. Dynamic M7 network graph (node size = centrality, edge width = spillover strength)
4. Spillover heatmap by quarter (showing NVDA's rise to center)
5. Time-varying total connectedness curve (with event annotations)
6. M0 -> M3 model performance comparison
7. Single event drill-down (e.g., NVDA 2024 Q2 earnings)
8. Simulated trading returns curve

Tech: Streamlit + Plotly

---

## Data Leakage Controls

- Sentiment data strictly cut off at ED-1
- ED day news completely excluded
- Correlation window [ED-90, ED-8] does not overlap with sentiment window
- Quiet period [ED-37, ED-30] far from earnings dates
- Walk-forward validation, never shuffle samples
- All features use only pre-ED data

---

## Academic References

- Diebold, F.X. and K. Yilmaz (2014). "On the Network Topology of Variance Decompositions: Measuring the Connectedness of Financial Firms." *Journal of Econometrics*, 182, 119-134.
- Nyakurukwa, K. and Y. Seetharam (2025). "Investor Sentiment Networks: Mapping Connectedness in DJIA Stocks." *Financial Innovation*, 11:4.

---

## Project Structure

```
hackathon2026/
├── README.md
├── .env                          # API keys (not committed)
├── papers/                       # Reference papers
├── data/
│   ├── earnings/
│   │   └── mag7_earnings_dates.csv   # 91 earnings events
│   ├── news_schedule/
│   │   ├── full_news_schedule.csv    # Complete fetch plan (9,555 records)
│   │   └── unique_fetch_list.csv     # Deduplicated fetch list (3,927 queries)
│   ├── news/                         # {TICKER}/{YEAR}/Q{N}/ (pending)
│   └── prices/                       # Stock price data (pending)
├── scripts/
│   ├── 01_build_news_schedule.py     # Generate earnings dates & fetch plan
│   └── 02_fetch_news.py             # Fetch news from EODHD API (pending rewrite)
└── notebooks/                        # Analysis notebooks (pending)
```

## Current Status

- [x] Project scope and pipeline design finalized
- [x] Earnings dates collected (91 events, 7 companies x 13 quarters)
- [x] News fetch schedule generated (3,927 unique company-date queries)
- [x] Spillover network methodology designed (Diebold-Yilmaz + Sentiment Connectedness)
- [ ] EODHD API integration for historical news
- [ ] Stock price data collection
- [ ] Sentiment scoring pipeline
- [ ] Spillover network computation (VAR + GFEVD)
- [ ] Feature engineering
- [ ] Model training and evaluation
- [ ] Dashboard / visualization
- [ ] Presentation
