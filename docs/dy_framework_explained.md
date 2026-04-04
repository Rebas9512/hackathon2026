# The Diebold-Yilmaz Spillover Network: Explained Like You're 3

> This document explains the core math behind our M3 spillover network features.
> No jargon walls. Every formula earns its place by being explained in plain words first.

---

## The Big Picture in One Sentence

We want to answer: **"When NVDA sneezes, does AAPL catch a cold?"** — and measure exactly how much of that cold came from NVDA vs. MSFT vs. TSLA vs. everyone else.

---

## Step 0: What Our Data Looks Like

We have two spreadsheets that run every day for ~4 years (2021-06 to 2025-03):

**Spreadsheet 1 — Daily Returns** (from `daily_prices.csv`):

| Date | AAPL | MSFT | GOOGL | AMZN | NVDA | META | TSLA |
|------|------|------|-------|------|------|------|------|
| Day 1 | +0.5% | +0.3% | -0.2% | +0.8% | +1.2% | +0.1% | -0.4% |
| Day 2 | +0.2% | +0.6% | +0.4% | +0.3% | +0.8% | +0.5% | +0.1% |
| Day 3 | -0.3% | -0.1% | -0.5% | -0.2% | -0.9% | -0.3% | -0.7% |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Spreadsheet 2 — Daily Sentiment** (from `daily_sentiment.csv`):

| Date | AAPL | MSFT | GOOGL | AMZN | NVDA | META | TSLA |
|------|------|------|-------|------|------|------|------|
| Day 1 | 0.74 | 0.81 | 0.62 | 0.55 | 0.90 | 0.43 | 0.32 |
| Day 2 | 0.59 | 0.77 | 0.58 | 0.60 | 0.85 | 0.50 | 0.28 |
| ... | ... | ... | ... | ... | ... | ... | ... |

For each earnings event, we slice a window of ~142 days and run the full pipeline below **twice** — once on returns, once on sentiment.

---

## Step 1: VAR — "Use Everyone's Yesterday to Predict Everyone's Today"

### The Intuition

Imagine 7 kids sitting in a circle, and every day each kid picks a mood (happy or grumpy). You notice patterns:

- When NVDA was happy yesterday, AAPL tends to be happy today.
- When TSLA was grumpy yesterday, META tends to be grumpy today.
- When AAPL was happy yesterday, it's often still happy today (momentum).

A **VAR model** is just writing these patterns down as equations. For **each** company, we write:

```
Today's AAPL = (some weight × Yesterday's AAPL)
             + (some weight × Yesterday's MSFT)
             + (some weight × Yesterday's GOOGL)
             + (some weight × Yesterday's AMZN)
             + (some weight × Yesterday's NVDA)
             + (some weight × Yesterday's META)
             + (some weight × Yesterday's TSLA)
             + random noise
```

We write 7 of these equations (one per company), each with 7 weights. That's **49 weights** total for VAR(1).

### What "VAR(p)" Means

The number in parentheses is how many days back we look:

- **VAR(1)**: only use yesterday → 7×7 = 49 weights
- **VAR(2)**: use yesterday AND the day before → 7×14 = 98 weights
- **VAR(3)**: three days back → 7×21 = 147 weights

More lags = more memory, but also more weights to estimate. With ~142 days of data and 7 variables, VAR(1) or VAR(2) is the sweet spot. We use BIC (a statistical score that penalizes complexity) to pick between them.

### How We Estimate the Weights

Plain old linear regression (OLS), one equation at a time. Nothing fancy. With 7 variables and ~140 observations, this is a very standard, well-behaved regression problem.

### The Math (For Reference)

A VAR(p) model in compact notation:

```
y_t = c + A₁·y_{t-1} + A₂·y_{t-2} + ... + Aₚ·y_{t-p} + u_t
```

Where:
- `y_t` is a 7×1 vector (today's values for all 7 companies)
- `A₁, A₂, ...` are 7×7 coefficient matrices (the "weights")
- `u_t` is a 7×1 noise vector with covariance matrix `Σ`
- `c` is a constant (intercept)

That's it. It's just multivariate linear regression.

---

## Step 2: The Thought Experiment — "What If NVDA Gets Surprised?"

Now the VAR model is estimated. We know all the weights. Time for a thought experiment.

### The Setup

Imagine it's a normal day. Suddenly, **NVDA gets a shock** — some unexpected news that moves it. Nobody else gets any direct shock. What happens next?

Because the VAR model says "today's MSFT partly depends on yesterday's NVDA", that NVDA shock starts **rippling through the system**:

```
Day 0:  NVDA gets shocked. Everyone else: nothing yet.
Day 1:  MSFT, AAPL, etc. move a little (because they partly depend on yesterday's NVDA).
Day 2:  The ripple continues — now companies react to Day 1's movements,
        which were themselves caused by NVDA's original shock.
Day 3:  Ripple gets weaker...
...
Day 10: Ripple is mostly gone.
```

This ripple is called an **Impulse Response Function (IRF)** — we "impulse" one variable and watch the "response" of all others.

### What We Actually Measure: Forecast Error Variance

Instead of tracking the ripple's direction (up or down), we track its **size** (how much wobbling it causes). Specifically:

> Over the next H days, AAPL's actual values will deviate from our VAR forecast.
> Some of that deviation is due to AAPL's own future shocks.
> Some is due to NVDA's original shock still rippling through.
> **What fraction of AAPL's total forecast wobble came from NVDA?**

That fraction is `d_{AAPL,NVDA}`. If it's 0.15, it means: "15% of the uncertainty in predicting AAPL over the next H days can be blamed on shocks originating from NVDA."

---

## Step 3: Why "Generalized"? — The Ordering Problem

### The Old Way (Cholesky): Order Matters

The original Diebold-Yilmaz (2009) method used something called Cholesky decomposition. The problem: you have to decide an **ordering** of the 7 companies. The results change depending on whether you put AAPL first or NVDA first.

Think of it like this: if two kids sneeze at the same time and a third kid gets sick, who do you blame? Cholesky says "blame whoever I listed first." That's arbitrary.

### The New Way (Pesaran-Shin 1998): Order Doesn't Matter

The Generalized approach (used in Diebold-Yilmaz 2012, 2014) does something smarter:

> For each company, pretend it's the ONLY one that got shocked.
> When computing NVDA's shock, allow that NVDA's shock might be slightly correlated with others (because in real life, shocks are correlated).
> But don't force an ordering — treat each company's shock experiment independently.

The trade-off: since we're running 7 independent experiments, the fractions for each row might not add up to exactly 100%. So we **normalize each row to sum to 1**:

```
d̃_{ij} = d_{ij} / Σⱼ d_{ij}
```

Now each row represents "what percentage of my forecast wobble came from each source?", and the percentages sum to 100%.

### The Math (For Reference)

The generalized H-step forecast error variance decomposition:

```
θ_{ij}(H) = σ_{jj}⁻¹ · Σ_{h=0}^{H-1} (e_i' · Ψ_h · Σ · e_j)²
            ÷ Σ_{h=0}^{H-1} (e_i' · Ψ_h · Σ · Ψ_h' · e_i)
```

Where:
- `Ψ_h` = the VAR's moving-average coefficient at horizon h (how shocks propagate h steps)
- `Σ` = the covariance matrix of VAR residuals (how correlated the noise terms are)
- `σ_{jj}` = variance of company j's noise (the diagonal element of Σ)
- `e_i` = selection vector (all zeros except a 1 in position i)

Then normalize:

```
d̃_{ij}(H) = θ_{ij}(H) / Σ_{j=1}^{N} θ_{ij}(H)
```

Don't worry about memorizing this. The key idea is: **shock company j → measure how much company i wobbles → that's the (i,j) entry of the matrix.**

---

## Step 4: The Connectedness Table = The Network

After running Step 3 for every pair (i, j), we get a **7×7 matrix**. This matrix IS the network.

### Reading the Matrix

```
"Who is affected"     ← AAPL   MSFT   GOOGL   AMZN   NVDA   META   TSLA   | FROM others
    ↓                    shock  shock  shock   shock  shock  shock  shock   | (In-Degree)
AAPL is affected by:    [0.42]  0.12   0.10    0.08   0.15   0.06   0.07   |  0.58
MSFT is affected by:     0.10  [0.38]  0.12    0.09   0.20   0.05   0.06   |  0.62
GOOGL is affected by:    0.08   0.13  [0.40]   0.11   0.16   0.07   0.05   |  0.60
AMZN is affected by:     0.09   0.10   0.11   [0.41]  0.14   0.08   0.07   |  0.59
NVDA is affected by:     0.06   0.08   0.07    0.05  [0.55]  0.04   0.15   |  0.45
META is affected by:     0.11   0.09   0.13    0.12   0.10  [0.36]  0.09   |  0.64
TSLA is affected by:     0.07   0.05   0.04    0.06   0.18   0.08  [0.52]  |  0.48
                        ──────────────────────────────────────────────────
TO others (Out-Degree):  0.51   0.57   0.57    0.51   0.93   0.38   0.49
```

**How to read this**:
- **Diagonal** `[0.42]`: AAPL explains 42% of its own wobble. The rest comes from others.
- **Row = "imports"**: reading across AAPL's row tells you who affects AAPL and by how much.
- **Column = "exports"**: reading down NVDA's column tells you how much NVDA affects everyone.
- **In-Degree** (row sum minus diagonal): total influence AAPL *receives* from all others = 0.58
- **Out-Degree** (column sum minus diagonal): total influence NVDA *exerts* on all others = 0.93

### Network Measures We Extract

| Measure | Formula | Plain English |
|---------|---------|---------------|
| **In-Degree** of i | `Σ_{j≠i} d̃_{ij}` | "How much do others push me around?" |
| **Out-Degree** of j | `Σ_{i≠j} d̃_{ij}` | "How much do I push others around?" |
| **Net Transmitter** of i | Out-Degree − In-Degree | Positive = you're a leader. Negative = you're a follower. |
| **System Connectedness** | `Σ_{i≠j} d̃_{ij} / N` | "How tightly glued together is the whole Mag7 system right now?" |

---

## Step 5: From One Matrix to M3 Features

### 5a: Dual-Layer Design

We run the entire pipeline (VAR → GFEVD → 7×7 matrix) **twice** for each earnings event:

| Layer | Input Data | What It Captures |
|-------|-----------|-----------------|
| **Return Network** | 7 companies' daily stock returns | Stock price co-movement and lead-lag relationships |
| **Sentiment Network** | 7 companies' daily sentiment scores | News narrative contagion — whose sentiment shifts predict whose? |

Then we combine:

```
W_{ij} = 0.5 × D_return_{ij} + 0.5 × D_sentiment_{ij}
```

Why two layers? Because a company might be tightly linked to another in **price** (they move together) but loosely linked in **sentiment** (their news cycles are independent), or vice versa. The dual layer captures both channels.

### 5b: The Four M3 Features

For each earnings event (company `c`, date `ED`), we extract from the composite network `W`:

**Feature 1: `spillover_weighted_sent`** — "What is the mood of the companies that influence me?"

```
spillover_weighted_sent_c = Σ_{j≠c} W_{cj} × sent_mean_j
```

This takes the sentiment of every other Mag7 company and weights it by how strongly that company's shocks spill into company c. If NVDA has high spillover weight into AAPL, and NVDA's pre-earnings sentiment is very positive, then AAPL's `spillover_weighted_sent` will be pulled upward.

**Feature 2: `net_transmitter`** — "Am I leading or following the pack right now?"

```
net_transmitter_c = Out-Degree_c − In-Degree_c
```

A high positive value means this company is a **net sender** of shocks (it pushes others around more than it gets pushed). NVDA during the AI boom would score high here. A negative value means the company is a **net receiver** (it's being dragged around by the group).

**Feature 3: `system_connectedness`** — "How tightly coupled is the entire Mag7 system?"

```
system_connectedness = Σ_{i≠j} W_{ij} / 7
```

This is the same number for all 7 companies at a given point in time. When it's high, the Mag7 is behaving like one giant stock — sentiment or price shocks in any one company ripple strongly to all others. This is the "system-wide AI hype" indicator.

**Why this matters for prediction**: When system connectedness is high AND a company has high pre-earnings sentiment, the "buy the rumor, sell the news" effect should be strongest — because the high sentiment isn't just company-specific, it's the entire ecosystem running hot.

**Feature 4: `spillover_neg_asym`** — "Am I getting hit by negative sentiment from others?"

```
spillover_neg_asym_c = Σ_{j≠c} W_{cj} × max(0, −sent_mean_j)
```

Only counts negative sentiment from other companies, weighted by spillover strength. This captures the asymmetry that Nyakurukwa & Seetharam (2025) documented: negative sentiment is more contagious than positive sentiment. If the companies that strongly influence you are all in a bad mood, that's a stronger signal than if they're all in a good mood.

---

## Step 6: The Timeline — What Data Goes Where

For one earnings event, say **AAPL 2024-08-01**:

```
                    ← 150 days →         ← 7d →  ED   ← 5d →
────[2024-01-15]────────────────[2024-07-22]───[08-01]───[08-08]───→

     |__________________________________|      |  |      |
     VAR window: [ED-150, ED-8]                |  |      |
     ~142 trading days of returns              |  |      |
     ~142 trading days of sentiment            |  |      |
                                               |  |      |
     Used to build the 7×7 network             |  |      Target:
                                               |  |      5-day return
                                               |  |      direction
                                        Sentiment |
                                        window    ED excluded
                                        [ED-7, ED-1]  (can't separate
                                        (M2 features)  pre/post news)
```

**No leakage**: the network is built entirely from data that was available *before* the sentiment window even begins. The network window ends at ED-8, the sentiment window starts at ED-7.

---

## Recap: The Full Pipeline for One Earnings Event

```
1. SLICE the data
   ├── Returns: 7 tickers × 142 days → a 142×7 matrix
   └── Sentiment: 7 tickers × 142 days → a 142×7 matrix

2. FIT two VAR models (one per layer)
   ├── VAR on returns → coefficient matrices A₁ (and maybe A₂)
   └── VAR on sentiment → coefficient matrices B₁ (and maybe B₂)

3. COMPUTE two GFEVD matrices (H=10 step forecast horizon)
   ├── D_return: 7×7 matrix of "who explains whose return wobble"
   └── D_sentiment: 7×7 matrix of "who explains whose sentiment wobble"

4. COMBINE into composite network
   └── W = 0.5 × D_return + 0.5 × D_sentiment

5. EXTRACT 4 features for the target company
   ├── spillover_weighted_sent
   ├── net_transmitter
   ├── system_connectedness
   └── spillover_neg_asym

6. APPEND to the feature matrix alongside M1 and M2 features
```

Repeat for all 91 earnings events. Done.

---

## References

- Diebold, F.X. and K. Yilmaz (2014). "On the Network Topology of Variance Decompositions." *Journal of Econometrics*, 182, 119-134.
- Diebold, F.X. and K. Yilmaz (2023). "On the Past, Present, and Future of the Diebold-Yilmaz Approach to Dynamic Network Connectedness." *Journal of Econometrics*, 50th Jubilee Issue.
- Nyakurukwa, K. and Y. Seetharam (2025). "Investor Sentiment Networks: Mapping Connectedness in DJIA Stocks." *Financial Innovation*, 11:4.
- Pesaran, M.H. and Y. Shin (1998). "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters*, 58, 17-29.
