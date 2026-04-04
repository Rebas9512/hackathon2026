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

## The Story: From Naive Rules to Sentiment Contrarian Signals

### Chapter 1 — M0: The Naive Baseline (Beat = Up, Miss = Down)

Our journey starts with the simplest possible model: if a company beats earnings expectations, predict the stock goes up; if it misses, predict down.

![M0 vs M1 Baseline Benchmark](outputs/m0_m1_baseline_benchmark.png)

| Metric | M0 (Beat/Miss Rule) | Random |
|--------|---------------------|--------|
| Accuracy | **0.549** | 0.500 |
| F1 | **0.676** | — |
| AUC-ROC | 0.557 | 0.500 |
| Gross Return | **+66.6%** | — |

This naive rule already tells us something: earnings surprises have *some* predictive power (+5% over random), but only weakly. Almost half the time, the stock moves in the opposite direction of what the surprise suggests. Why?

### Chapter 2 — M1: Can Surprise Magnitude Help?

Instead of just beat/miss, M1 uses logistic regression on the actual EPS surprise percentage — maybe the *size* of the beat matters.

| Metric | M0 (Beat/Miss) | M1 (LogReg on Surprise %) | Random |
|--------|----------------|---------------------------|--------|
| Accuracy | **0.549** | 0.490 | 0.500 |
| F1 | **0.676** | 0.658 | — |
| AUC-ROC | 0.557 | **0.634** | 0.500 |
| Gross Return | **+66.6%** | +43.1% | — |

**Finding**: M1 has better AUC (0.634), confirming that larger surprises do correlate with positive returns. But it can't find a useful decision boundary — the signal is too weak with just one feature. M1 ends up predicting *everything* as "up" (= Buy & Hold), with its return curve overlapping M1's at +43.1%.

**Takeaway**: Earnings surprise alone is not enough. The market clearly prices in *something else* before earnings day.

### Chapter 3 — M2: Adding Pre-Earnings Sentiment

Could that "something else" be sentiment? M2 adds 4 features computed from 7-day pre-earnings news:

- `sent_mean`: average article polarity in [ED-7, ED-1]
- `sent_trend`: sentiment slope (late vs early in the window)
- `sent_delta`: sentiment anomaly vs quiet period baseline
- `news_volume`: total articles in the window

![M0 vs M1 vs M2 Comparison](outputs/m0_m1_m2_comparison.png)

| Metric | M0 (Beat/Miss) | M1 (Surprise %) | M2 (+ Sentiment) | Random |
|--------|----------------|------------------|-------------------|--------|
| Accuracy | 0.542 | 0.419 | 0.500 | 0.500 |
| F1 | 0.667 | 0.648 | 0.600 | — |
| AUC-ROC | 0.558 | 0.617 | 0.447 | 0.500 |
| Gross Return | +24.3% | +6.8% | **+30.2%** | — |

> Note: All three models evaluated on the same reduced test set (48 events with complete sentiment data) for fair comparison.

A paradox emerges: **M2's AUC (0.447) is worse than random**, yet it achieves the **best trading return (+30.2%)**. How?

All 4 sentiment features have **negative coefficients** — the model learned that high pre-earnings sentiment is a *sell signal*. This is the classic **"buy the rumor, sell the news"** dynamic: when pre-earnings hype is already high, the good news is priced in, and the stock drops even on a beat.

---

## Chapter 4 — Deep Dive: M2's Victories and the "Beat But Drop" Pattern

To understand when and why sentiment adds value, we dissected every event where M2 disagreed with M0.

![M2 Sentiment vs Surprise](outputs/m2_sentiment_vs_surprise.png)

### M2 Won 6 Events, Lost 8 — But Won Bigger

| | Events | Total PnL Impact |
|--|--------|-----------------|
| M2 victories over M0 | 6 | — |
| M2 failures vs M0 | 8 | -108.8% cumulative |
| **Net M2 edge over M0** | — | **+3.6%** |

M2 wins fewer events but captures larger moves — its victories came on high-magnitude drops and one massive rally.

![M2 PnL Attribution](outputs/m2_pnl_attribution.png)

### The 5 "Beat But Drop" Victories

M2 correctly predicted **DOWN** despite positive EPS surprise in 5 events. The pattern: very high pre-earnings sentiment (0.67–0.75) + EPS beat → stock drops anyway.

---

#### AAPL — 2023-08-03

> EPS Surprise: **+5.7%** (BEAT) | 5-Day Return: **-6.9%** (DROPPED)  
> Pre-earnings sentiment: 0.687 | 133 articles

**What happened**: Apple beat on EPS, driven by services revenue surpassing 1 billion users. But the headline story was a *third consecutive quarter of declining sales*, with iPhone demand slumping. The "services pivot" narrative was already priced in — investors focused on hardware weakness.

Key headlines:
| Polarity | Headline |
|----------|----------|
| -0.973 | Apple and Amazon to report, Adidas narrows loss forecast - what's moving markets |
| -0.878 | Trump arraigned in D.C., Amazon earnings, Apple's declining revenue: Top Stories |
| -0.823 | Russia fines Apple for not deleting 'inaccurate' content on Ukraine conflict |
| -0.402 | Apple Earnings Preview: Can Services Revenue and AI Hopes Offset Weaker iPhone Demand? |
| +1.000 | Apple earnings beat estimates, services boost results |

---

#### AAPL — 2024-08-01

> EPS Surprise: **+4.3%** (BEAT) | 5-Day Return: **-2.3%** (DROPPED)  
> Pre-earnings sentiment: 0.685 | 122 articles

**What happened**: Another Apple beat, but the market reacted to broader tech weakness. Nvidia tumbled on the same day, and macro fears (jobless claims) overshadowed individual earnings beats. The antitrust lawsuit narrative added negative pressure.

Key headlines:
| Polarity | Headline |
|----------|----------|
| -0.835 | Apple Says US Smartphones Suit Has 'No Relation to Reality' |
| -0.778 | Apple asks US judge to toss antitrust lawsuit |
| -0.361 | Dow Jones Futures Fall As Apple, Amazon Follow Market Expectations Breaker; Nvidia Tumbles |
| +0.999 | 4 Top Tech Stocks to Buy on Soaring Hopes of September Rate Cut |

---

#### MSFT — 2025-01-29

> EPS Surprise: **+4.1%** (BEAT) | 5-Day Return: **-6.6%** (DROPPED)  
> Pre-earnings sentiment: 0.748 (highest in the dataset) | 118 articles

**What happened**: This was the **DeepSeek shock** week. Microsoft beat earnings, but the market was reeling from the DeepSeek AI breakthrough — which threatened to undermine the massive AI infrastructure spending that justified Microsoft's (and the entire Mag7's) valuations. Pre-earnings sentiment was at 0.748, the highest of any event, reflecting peak AI optimism that DeepSeek abruptly punctured.

Key headlines:
| Polarity | Headline |
|----------|----------|
| -0.869 | Alibaba Surges as AI Battle Heats Up; Nvidia Rebounds After Historic Loss |
| -0.542 | DeepSeek AI Fears Haunt Meta, Microsoft Earnings. What Stock Markets Need to See |
| -0.495 | Tech earnings ahead, Fed decision, ASML reports - what's moving markets |
| +0.000 | Market Chatter: Microsoft, OpenAI Find Evidence DeepSeek Breached Rules in Developing AI Models |
| +1.000 | Mag 7 Earnings: Tesla, Microsoft & Meta in Focus |

---

#### GOOGL — 2025-02-04

> EPS Surprise: **+1.2%** (BEAT) | 5-Day Return: **-10.2%** (DROPPED, largest magnitude)  
> Pre-earnings sentiment: 0.732 | 123 articles

**What happened**: Alphabet beat narrowly, but the earnings landed in the middle of the **US-China trade war escalation** (Trump tariffs → China retaliation). Cloud growth concerns compounded the macro pressure. The decision to reverse the ban on AI for weapons (following Palantir) added controversy. Sentiment at 0.732 was sky-high, but the geopolitical reality crushed the stock.

Key headlines:
| Polarity | Headline |
|----------|----------|
| -0.962 | Gold steady near record high as Trump starts US-China trade war |
| -0.836 | AMD, Alphabet fall after Q4 results: Biggest earnings takeaways |
| -0.652 | China Hits Back Against Trump's Tariffs With Targeted Actions |
| -0.494 | Alphabet reverses ban on AI use for weapons, following Palantir |
| +0.999 | Alphabet's Q4 Earnings: Revenue In Line With Expectations But Stock Drops |

---

#### AMZN — 2025-02-06

> EPS Surprise: **+25.4%** (BEAT, massive) | 5-Day Return: **-3.5%** (DROPPED)  
> Pre-earnings sentiment: 0.675 | 152 articles

**What happened**: Amazon crushed earnings by 25% — but even that wasn't enough. The Shein/Temu tariff story hit the same day (30% levy on US-bound goods), threatening Amazon's competitive position. Meanwhile, cloud growth fears persisted after Microsoft and Google both disappointed on cloud metrics. The highest article count (152) in any window reflected peak attention, but the stock dropped anyway.

Key headlines:
| Polarity | Headline |
|----------|----------|
| -0.975 | Shein, Temu Retailers Slapped With 30% Levy on US-Bound Goods |
| +0.178 | Amazon Earnings Due Today. Cloud Growth Is In Focus After Microsoft, Google Stumbles |
| +0.477 | Amazon EPS Jumps 86%, Beats Forecasts |
| +0.659 | Lyft Joins Amazon and Anthropic to Revolutionize AI Customer Service |

---

### The 1 "Miss But Rally" Victory: TSLA 2024-04-23

> EPS Surprise: **-8.1%** (MISS) | 5-Day Return: **+26.7%** (MASSIVE RALLY)  
> Pre-earnings sentiment: 0.323 (lowest in dataset) | Sentiment trend: -0.103 | Sentiment delta: -0.219 | 186 articles

This single event accounts for the **majority of M2's PnL edge** — a +53.4% swing vs M0.

**What happened**: Tesla was in crisis mode before earnings. The stock had fallen 43%, Cybertruck reviews were devastating, mass layoffs in Germany/Texas/California, and headlines screamed "Disaster at Tesla." Sentiment was at 0.323 — the most negative in the entire dataset, with a steep downward trend (-0.103) and a massive negative delta from quiet period (-0.219).

Then earnings missed *as expected* — the bad news was fully priced in. The market pivoted to the forward-looking announcement: **Tesla would accelerate the rollout of cheaper electric cars**. The stock surged 26.7% in 5 days.

M2 saw what M0 couldn't: when sentiment is this negative and expectations are this low, even a miss becomes a clearing event.

Key headlines on earnings day:
| Polarity | Headline |
|----------|----------|
| -0.976 | Tesla Stock in 'No Man's Land' After 43% Rout Ahead of Earnings |
| -0.966 | Disaster at Tesla? Previewing Today's Earnings |
| -0.960 | I Was Incredibly Wrong About the Tesla Cybertruck |
| -0.827 | Tesla aims to cut 400 jobs in Germany via voluntary programme |
| -0.296 | TSLA Stock Earnings: Tesla Misses EPS, Misses Revenue for Q1 2024 |
| +0.911 | Tesla to speed up rollout of cheaper electric cars |
| +0.937 | Tesla's Q1 revenue falls 9% amid mass layoffs in California, Texas |
| +0.984 | General Motors beats quarterly results targets, raises forecast |
| +0.997 | Elon Musk plots new direction after Tesla's electric car crisis |

---

### M2's Biggest Failure: NVDA 2024-02-21

> EPS Surprise: **+11.4%** (BEAT) | 5-Day Return: **+15.1%** (RALLIED)  
> Sentiment: 0.63 | M0: Up (correct) | M2: Down (wrong)

M2's contrarian signal said: high sentiment + beat = sell. But NVDA was in the **middle of the AI infrastructure supercycle** — this wasn't priced-in hype, it was genuine fundamental momentum. The model couldn't distinguish company-specific justified optimism from system-wide irrational exuberance.

This is exactly the problem M3's cross-company spillover features are designed to solve.

---

## Key Insight: Why M3 Matters

The "buy the rumor, sell the news" pattern that M2 discovered is strongest when **system-wide AI hype** is elevated — not just company-specific sentiment. M2 fails when it can't distinguish:

- **High sentiment in a generally hyped market** → likely priced in → sell (MSFT Jan 2025, GOOGL Feb 2025)
- **High sentiment with peers at neutral/low** → company-specific good news → may still have upside (NVDA Feb 2024)

M3's cross-company spillover network features (from the Diebold-Yilmaz framework) are designed to capture exactly this distinction by measuring how tightly coupled the Mag7 sentiment system is at any given time.

---

## Three-Model Progressive Comparison

| Model | Features | Purpose |
|-------|----------|---------|
| **M0: Naive Rule** | Beat → Up, Miss → Down | "Does the surprise direction matter?" |
| **M1: Baseline** | Logistic regression on surprise % | "Does the surprise magnitude help?" |
| **M2: + Sentiment** | + 4 pre-earnings sentiment features | "Does sentiment add predictive power?" |
| **M3: + Spillover** | + dual-layer DY network cross-company signals | "Does cross-company context help?" |

---

## Pipeline Design

### Phase 1: Data Collection (Complete)

| Data | Source | Format | Status |
|------|--------|--------|--------|
| Daily stock prices (OHLCV) | yfinance | M7+SPY, 962 trading days (2021-06 ~ 2025-03) | Done |
| Earnings dates + EPS surprise | yfinance | 91 events, 100% EPS coverage | Done |
| Daily aggregated sentiment | EODHD `/api/sentiments` | 7 tickers x ~1,300 days continuous (2021-06 ~ 2025-03) | Done |
| Pre-earnings window news | EODHD `/api/news` | 13,588 articles in [ED-7, ED-1] with per-article scores | Done |
| ED-day news | EODHD `/api/news` | 563 articles on earnings dates for post-hoc analysis | Done |

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

**M1 - Baseline Features**
- `surprise_pct`: actual vs consensus EPS

**M2 - Direct Sentiment Features (company-specific, 7-day window)**

Computed from article-level data:
- `sent_mean`: average polarity over [ED-7, ED-1] articles
- `sent_trend`: sentiment slope (later 3 days mean - earlier 4 days mean)
- `news_volume`: total number of articles in sentiment window

Computed using daily-aggregate data for quiet period:
- `sent_delta`: sent_mean - quiet_period_mean (anomaly signal vs [ED-37, ED-30] baseline)

**M3 - Spillover Network Features (cross-company, dual-layer DY framework)**
- `spillover_weighted_sent`: other companies' sentiment weighted by composite network
- `net_transmitter`: out_degree - in_degree (sender or receiver of shocks?)
- `system_connectedness`: total off-diagonal sum / N (how tightly coupled is the M7 system?)
- `spillover_neg_asym`: weighted sum of spillover from companies with negative sentiment

### Phase 5: Dual-Layer Spillover Network — Diebold-Yilmaz Framework (M3 Core)

The key innovation: with continuous daily sentiment data for all 7 companies, we can build **dual-layer** dynamic connectedness networks — one for returns and one for sentiment — following the Diebold-Yilmaz (2014) framework.

#### Layer 1: Return Connectedness

```
VAR(p) on M7 daily returns → Generalized FEVD (Pesaran-Shin 1998)
→ d_ij_return = fraction of i's return forecast error variance explained by j's shocks
→ Weighted, directed return spillover network
```

#### Layer 2: Sentiment Connectedness

```
VAR(p) on M7 daily sentiment → Generalized FEVD
→ d_ij_sentiment = "whose sentiment change predicts whose?"
→ Captures sentiment contagion structure
```

#### Composite Spillover Weight

```
W_ij = α × d_ij_return + β × d_ij_sentiment
```

### Phase 6: Modeling & Evaluation

**Algorithms:** Logistic Regression (primary, 91 samples → simple models preferred), XGBoost (robustness check)

**Validation:** Time-ordered expanding window walk-forward (train on first T events → predict T+1 → expand, start after ~40 events)

**Metrics:** Accuracy, AUC-ROC, F1, Simulated trading returns, McNemar test for model differences

### Phase 7: Visualization / Dashboard

Tech: Streamlit + Plotly

1. Sentiment timeline vs stock price overlay
2. Pre-earnings sentiment vs quiet period comparison
3. Dynamic M7 network graph (interactive, by quarter)
4. Spillover heatmap by quarter
5. Time-varying total connectedness curve
6. M0 → M1 → M2 → M3 model comparison
7. Single event drill-down
8. Simulated trading returns curve

---

## Data Leakage Controls

- Sentiment data strictly cut off at ED-1
- ED day news completely excluded from features (only used for post-hoc analysis)
- DY network window [ED-150, ED-8] does not overlap with sentiment window [ED-7, ED-1]
- Quiet period [ED-37, ED-30] far from both sentiment window and earnings date
- Walk-forward validation, never shuffle samples
- All features use only pre-ED data

---

## Academic References

- Diebold, F.X. and K. Yilmaz (2014). "On the Network Topology of Variance Decompositions." *Journal of Econometrics*, 182, 119-134.
- Nyakurukwa, K. and Y. Seetharam (2025). "Investor Sentiment Networks." *Financial Innovation*, 11:4.
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
│   ├── hackathon.db                      # SQLite database (all tables, gitignored)
│   ├── earnings/
│   │   └── mag7_earnings.csv             # 91 events + EPS surprise
│   ├── prices/
│   │   └── daily_prices.csv              # M7+SPY daily OHLCV, 2021-06~2025-03
│   ├── sentiment/
│   │   ├── daily_sentiment.csv           # Daily aggregated sentiment, 2021-06~2025-03
│   │   └── extreme_events.csv           # Notable sentiment events for dashboard
│   └── news/
│       ├── window_articles.csv           # 13,588 articles in [ED-7,ED-1] windows
│       └── ed_day_articles.csv           # 563 ED-day articles for post-hoc analysis
├── scripts/
│   ├── 01_fetch_sentiment.py
│   ├── 02_fetch_prices.py
│   ├── 03_fetch_window_news.py
│   ├── build_db.py
│   └── clean_data.py
├── notebooks/
│   ├── 01_baseline_benchmark.ipynb       # M0 vs M1 baselines
│   ├── 02_m2_sentiment.ipynb            # M2 sentiment model + comparison
│   └── 03_m2_deep_dive.ipynb            # M2 victory/failure analysis + news
├── outputs/                              # Charts and figures
└── app/                                 # Streamlit dashboard (pending)
```

---

## Current Status

- [x] Data collection complete (prices, EPS, sentiment, window news, ED-day news)
- [x] M0 baseline: naive beat/miss rule (accuracy 0.549, return +66.6%)
- [x] M1 baseline: logistic regression on surprise % (AUC 0.634, but predicts all-up)
- [x] M2 evaluated: + sentiment features (AUC 0.447, return +30.2%)
- [x] M2 deep dive: identified "buy the rumor, sell the news" pattern in 5 events + TSLA contrarian rally
- [ ] DY spillover network computation (or correlation fallback)
- [ ] M3 evaluation (+ spillover network features)
- [ ] Dashboard / visualization
- [ ] Presentation
