"""
Spillover Alpha — Interactive Dashboard
2026 FinHack Challenge, Case 4

Multi-source sentiment spillover network meets post-earnings prediction.
Single-page terminal-style dashboard:
  Top bar:   ticker tabs (ALL + 7 Mag7)
  Left:      stock price chart + model overlays + cumulative return
  Right:     model detail panel + event explorer
  Bottom:    model toggle buttons
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_all_data, get_model_coefficients, M3_FEATURES, M2_FEATURES
from charts import (
    build_price_chart, build_cumulative_return_chart,
    build_network_graph, build_network_dynamic, build_network_animated,
    COLORS, MODEL_COLORS, MODEL_NAMES,
)

# ── Page Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spillover Alpha",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global dark theme */
    .stApp { background-color: #0A0A0A; }
    header[data-testid="stHeader"] { background-color: #0A0A0A; }
    .block-container { padding: 0.5rem 1rem 1rem 1rem; max-width: 100%; }

    /* Hide default streamlit elements */
    #MainMenu, footer, .stDeployButton { display: none; }

    /* Typography */
    h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif !important; }
    p, span, div, label { font-family: 'Inter', -apple-system, sans-serif; }
    code { font-family: 'IBM Plex Mono', monospace; }

    /* Card containers */
    .card {
        background: #111114; border: 1px solid #27272A; border-radius: 6px;
        padding: 16px; margin-bottom: 8px;
    }
    .card-dark { background: #0D0D10; border: 1px solid #27272A; border-radius: 6px; padding: 12px; }

    /* Metric display */
    .metric-value { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; }
    .metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px;
                    letter-spacing: 2px; color: #71717A; text-transform: uppercase; }
    .metric-green { color: #22C55E; }
    .metric-red { color: #EF4444; }
    .metric-purple { color: #A855F7; }
    .metric-blue { color: #3B82F6; }
    .metric-yellow { color: #F59E0B; }

    /* Feature bar */
    .feat-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
    .feat-name { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
                 color: #A1A1AA; min-width: 170px; }
    .feat-val { font-family: 'IBM Plex Mono', monospace; font-size: 11px; min-width: 50px; }
    .bar-neg { background: #EF4444; height: 12px; border-radius: 2px; }
    .bar-pos { background: #22C55E; height: 12px; border-radius: 2px; }

    /* Insight box */
    .insight-box {
        background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 6px; padding: 14px; margin: 8px 0;
    }
    .insight-title { font-family: 'IBM Plex Mono', monospace; font-size: 10px;
                     letter-spacing: 2px; color: #A855F7; margin-bottom: 6px; }
    .insight-text { font-family: 'Inter', sans-serif; font-size: 12px;
                    color: #A1A1AA; line-height: 1.6; }

    /* Event card */
    .event-card {
        background: #18181B; border-radius: 6px; padding: 14px; margin-top: 8px;
    }
    .event-header { display: flex; justify-content: space-between; align-items: center; }
    .event-ticker { font-family: 'Inter', sans-serif; font-size: 14px;
                    color: #FFF; font-weight: 600; }
    .event-badge { font-family: 'IBM Plex Mono', monospace; font-size: 10px;
                   padding: 2px 8px; border-radius: 4px; font-weight: 600; }

    /* Ticker tabs active state */
    .stRadio > div { flex-direction: row !important; gap: 4px !important; }
    .stRadio > div > label {
        background: #27272A !important; border-radius: 4px !important;
        padding: 4px 14px !important; color: #A1A1AA !important;
        font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #27272A; border-radius: 4px; padding: 6px 14px;
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #A1A1AA;
    }
    .stTabs [aria-selected="true"] {
        background: #A855F7 !important; color: #FFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Divider */
    hr { border-color: #27272A !important; margin: 8px 0 !important; }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        border: 1px solid #27272A; background: #27272A; color: #A1A1AA;
        border-radius: 4px; padding: 4px 16px;
    }
    .stButton > button:hover { border-color: #A855F7; color: #A855F7; }

    /* Plotly chart container */
    .stPlotlyChart { border: 1px solid #27272A; border-radius: 6px; overflow: hidden; }

    /* Columns gap reduction */
    [data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    /* Multiselect pills */
    .stMultiSelect [data-baseweb="tag"] {
        background: #A855F71A; border: 1px solid #A855F7AA;
    }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 10px;
        letter-spacing: 2.5px; color: #A855F7; margin-bottom: 4px;
    }

    /* Prediction badge */
    .pred-correct { background: rgba(34,197,94,0.1); color: #22C55E;
                    padding: 2px 8px; border-radius: 4px; font-size: 11px;
                    font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    .pred-wrong { background: rgba(239,68,68,0.1); color: #EF4444;
                  padding: 2px 8px; border-radius: 4px; font-size: 11px;
                  font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

    /* News headline */
    .headline-row { display: flex; gap: 10px; align-items: baseline; padding: 3px 0;
                    border-bottom: 1px solid #27272A22; }
    .headline-polarity { font-family: 'IBM Plex Mono', monospace; font-size: 11px;
                         min-width: 50px; }
    .headline-text { font-family: 'Inter', sans-serif; font-size: 12px;
                     color: #A1A1AA; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data & running walk-forward predictions...")
def cached_load():
    prices, events_df, snapshots = load_all_data()
    coefficients = get_model_coefficients(events_df)
    return prices, events_df, snapshots, coefficients

prices, events_df, snapshots, coefficients = cached_load()

# ── Session State ─────────────────────────────────────────────────────────

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"
if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["M3"]
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "active_chart_model" not in st.session_state:
    st.session_state.active_chart_model = None


# ── Model Info Database ───────────────────────────────────────────────────

MODEL_INFO = {
    "M0": {
        "name": "M0: Naive Rule",
        "desc": "If EPS actual > consensus → predict UP, else DOWN. No training needed.",
        "features": ["surprise_pct (binary threshold)"],
        "insight": "Earnings surprises have weak predictive power. Mag7 beats 90% of the time, so M0 almost always predicts UP — high recall but low precision.",
        "color": "#6B7280",
    },
    "M1": {
        "name": "M1: Surprise Magnitude",
        "desc": "Logistic regression on surprise %. Tests whether the size of the beat matters.",
        "features": ["surprise_pct"],
        "insight": "Larger surprises correlate with positive returns (AUC improves) but can't find a useful decision boundary. The market prices in something beyond the number itself.",
        "color": "#6B7280",
    },
    "M2": {
        "name": "M2: + Sentiment",
        "desc": "Adds 4 pre-earnings sentiment features from 7-day news window [ED-7, ED-1].",
        "features": M2_FEATURES,
        "insight": "All sentiment coefficients are NEGATIVE — high pre-earnings sentiment is a sell signal. The classic 'buy the rumor, sell the news' pattern. AUC drops below random, but trading return improves.",
        "color": "#3B82F6",
    },
    "M3": {
        "name": "M3: + Spillover Network",
        "desc": "Adds 4 Diebold-Yilmaz dual-layer VAR-GFEVD cross-company spillover features.",
        "features": M3_FEATURES,
        "insight": "spillover_weighted_sent is the strongest feature (coeff -0.826): when influential peers are all bullish, that's a network-level sell signal. system_connectedness acts as a regime switch.",
        "color": "#A855F7",
    },
    "XGB": {
        "name": "M3 XGBoost (depth=2)",
        "desc": "Same 9 features as M3, but XGBoost learns non-linear regime thresholds.",
        "features": M3_FEATURES,
        "insight": "Captures interaction effects: 'high connectedness AND high sentiment → strong sell' and 'low connectedness AND positive surprise → strong buy'. +205.9% return but caveat: 29 test events.",
        "color": "#F59E0B",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

# ── Top Bar: Logo + Ticker Tabs ───────────────────────────────────────────

st.markdown(
    '<h1 style="font-size:22px;margin:0 0 4px 0;padding:4px 0;color:#FFF;">'
    '⚡ Spillover Alpha'
    '<span style="font-size:12px;color:#71717A;font-weight:400;margin-left:16px;">'
    'Sentiment Spillover × Post-Earnings Prediction</span></h1>',
    unsafe_allow_html=True
)

tickers = ["ALL", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
ticker_cols = st.columns(len(tickers))
for i, tk in enumerate(tickers):
    with ticker_cols[i]:
        is_active = st.session_state.selected_ticker == tk
        if st.button(
            tk, key=f"tk_{tk}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.selected_ticker = tk
            st.rerun()

st.markdown('<hr style="margin:2px 0 4px 0;">', unsafe_allow_html=True)

# ── Main Body: Chart (left) + Detail Panel (right) ───────────────────────

chart_col, detail_col = st.columns([2.2, 1], gap="small")

# ── LEFT: Chart Area ─────────────────────────────────────────────────────

with chart_col:
    ticker = st.session_state.selected_ticker

    # ── Chart header ──
    if ticker == "ALL":
        st.markdown(
            '<div style="padding:4px 0;">'
            '<span style="font-family:Playfair Display,serif;font-size:28px;color:#FFF;font-weight:700;">'
            'Magnificent 7 — Normalized Returns</span></div>',
            unsafe_allow_html=True
        )
    else:
        latest = prices[prices["ticker"] == ticker].sort_values("date").iloc[-1]
        prev = prices[prices["ticker"] == ticker].sort_values("date").iloc[-2]
        change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100
        color = COLORS["green"] if change_pct >= 0 else COLORS["red"]
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:16px;padding:4px 0;">'
            f'<span style="font-family:Playfair Display,serif;font-size:28px;color:#FFF;font-weight:700;">{ticker}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:18px;color:#FFF;">${latest["close"]:.2f}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:13px;color:{color};font-weight:600;">{change_pct:+.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Main chart ──
    # Price chart only shows the explicitly active model's buy/sell markers
    chart_overlay = [st.session_state.active_chart_model] if st.session_state.active_chart_model else []
    fig = build_price_chart(
        prices, events_df, ticker,
        chart_overlay,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Model toggle bar (shared by ALL and single ticker) ──
    st.markdown(
        f'<div class="section-label">{"ALL STRATEGIES — CUMULATIVE RETURN" if ticker == "ALL" else "MODEL STRATEGY OVERLAY"}</div>',
        unsafe_allow_html=True
    )
    model_cols = st.columns(5)
    all_models = ["M0", "M1", "M2", "M3", "XGB"]

    for i, model_key in enumerate(all_models):
        with model_cols[i]:
            in_list = model_key in st.session_state.selected_models
            is_chart_active = st.session_state.active_chart_model == model_key
            label_prefix = "● " if is_chart_active else ("◆ " if in_list else "○ ")
            if st.button(
                f"{label_prefix}{MODEL_NAMES[model_key]}",
                key=f"btn_{model_key}",
                use_container_width=True,
            ):
                if in_list:
                    st.session_state.selected_models.remove(model_key)
                    if st.session_state.active_chart_model == model_key:
                        st.session_state.active_chart_model = None
                else:
                    st.session_state.selected_models.append(model_key)
                    st.session_state.active_chart_model = model_key
                st.rerun()

    # ── Cumulative return chart ──
    st.markdown('<div class="section-label" style="margin-top:8px;">CUMULATIVE STRATEGY RETURN</div>',
                unsafe_allow_html=True)
    if st.session_state.selected_models:
        cum_fig = build_cumulative_return_chart(
            events_df, st.session_state.selected_models, ticker
        )
        st.plotly_chart(cum_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        cum_fig = build_cumulative_return_chart(events_df, [], ticker)
        st.plotly_chart(cum_fig, use_container_width=True, config={"displayModeBar": False})


# ── RIGHT: Model Detail Panel ─────────────────────────────────────────────

with detail_col:
    active_model = st.session_state.active_chart_model
    # If no active model, show summary of all; otherwise show single model detail
    show_summary = active_model is None

    # ── Compute scoped test set (same logic as cumulative chart) ──
    all_test = events_df[events_df["is_test"]].copy()
    if ticker != "ALL":
        all_test = all_test[all_test["ticker"] == ticker]

    sel = st.session_state.selected_models
    has_xgb_or_m3 = any(m in sel for m in ["XGB", "M3"])
    has_m2 = "M2" in sel

    if has_xgb_or_m3 or (active_model in ["XGB", "M3"]):
        scoped_test = all_test[all_test["m3_pred"].notna()].copy()
        scope_tag = f"{len(scoped_test)}-event spillover set"
    elif has_m2 or active_model == "M2":
        scoped_test = all_test[all_test["m2_pred"].notna()].copy()
        scope_tag = f"{len(scoped_test)}-event sentiment set"
    else:
        scoped_test = all_test.copy()
        scope_tag = f"{len(scoped_test)}-event full set"

    def _calc_metrics(df, pred_col):
        v = df[df[pred_col].notna()]
        if v.empty:
            return None
        yt = v["target"].values
        yp = v[pred_col].values.astype(int)
        acc = (yt == yp).mean()
        sr = np.where(yp == 1, v["ret_5d"], -v["ret_5d"])
        gr = (1 + sr).prod() - 1
        return {"acc": acc, "ret": gr, "n": len(v)}

    if show_summary:
        # ── Summary view: all models on scoped test set ──
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
            '<span class="metric-label">MODEL OVERVIEW</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">{scope_tag}</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<hr>', unsafe_allow_html=True)

        for mk in ["M0", "M1", "M2", "M3", "XGB"]:
            info = MODEL_INFO[mk]
            m = _calc_metrics(scoped_test, f"{mk.lower()}_pred")
            if m is None:
                continue
            ret_color = "#22C55E" if m["ret"] > 0 else "#EF4444"

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
                f'margin:4px 0;background:#18181B;border-left:3px solid {info["color"]};border-radius:0 4px 4px 0;">'
                f'<div style="min-width:70px;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{info["color"]};font-weight:600;">{mk}</div>'
                f'<div style="font-family:Inter,sans-serif;font-size:11px;color:#A1A1AA;">{info["name"].split(":")[0] if ":" in info["name"] else info["name"]}</div>'
                f'</div>'
                f'<div style="flex:1;text-align:center;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:#FFF;font-weight:600;">{m["acc"]:.0%}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">ACC</div>'
                f'</div>'
                f'<div style="flex:1;text-align:center;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:{ret_color};font-weight:600;">{m["ret"]:+.1%}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">RETURN</div>'
                f'</div>'
                f'<div style="flex:1;text-align:center;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:#A1A1AA;">{m["n"]}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">EVENTS</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="insight-box">'
            '<div class="insight-title">SELECT A MODEL</div>'
            '<div class="insight-text">Click a model button on the left to see its buy/sell signals on the price chart and detailed feature analysis here.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:
        # ── Single model detail view (on scoped test set) ──
        info = MODEL_INFO[active_model]

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
            f'<span class="metric-label">MODEL DETAILS</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{info["color"]};font-weight:600;">● {active_model}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;margin-bottom:4px;">{scope_tag}</div>',
            unsafe_allow_html=True
        )
        st.markdown('<hr>', unsafe_allow_html=True)

        st.markdown(f'<div class="card">'
                    f'<div style="font-size:15px;color:#FFF;font-weight:600;margin-bottom:6px;">{info["name"]}</div>'
                    f'<div style="font-size:12px;color:#A1A1AA;line-height:1.5;">{info["desc"]}</div>'
                    f'</div>', unsafe_allow_html=True)

        m = _calc_metrics(scoped_test, f"{active_model.lower()}_pred")
        if m:
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f'<div class="card" style="text-align:center;">'
                         f'<div class="metric-value metric-purple">{m["acc"]:.1%}</div>'
                         f'<div class="metric-label">Accuracy</div></div>',
                         unsafe_allow_html=True)
            mc2.markdown(f'<div class="card" style="text-align:center;">'
                         f'<div class="metric-value metric-purple">{m["n"]}</div>'
                         f'<div class="metric-label">Events</div></div>',
                         unsafe_allow_html=True)
            ret_color = "metric-green" if m["ret"] > 0 else "metric-red"
            mc3.markdown(f'<div class="card" style="text-align:center;">'
                         f'<div class="metric-value {ret_color}">{m["ret"]:+.1%}</div>'
                         f'<div class="metric-label">Return</div></div>',
                         unsafe_allow_html=True)

        if active_model in ["M2", "M3", "XGB"]:
            st.markdown('<div class="metric-label" style="margin:8px 0 4px;">TOP FEATURES</div>',
                        unsafe_allow_html=True)
            feat_list = M3_FEATURES if active_model in ["M3", "XGB"] else M2_FEATURES
            max_abs = max(abs(v) for k, v in coefficients.items() if k in feat_list) if coefficients else 1
            for feat in feat_list:
                val = coefficients.get(feat, 0)
                bar_width = int(abs(val) / max_abs * 120)
                bar_class = "bar-neg" if val < 0 else "bar-pos"
                val_color = "#EF4444" if val < 0 else "#22C55E"
                st.markdown(
                    f'<div class="feat-bar">'
                    f'<span class="feat-name">{feat}</span>'
                    f'<div class="{bar_class}" style="width:{bar_width}px;"></div>'
                    f'<span class="feat-val" style="color:{val_color};">{val:+.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            f'<div class="insight-box">'
            f'<div class="insight-title">KEY INSIGHT</div>'
            f'<div class="insight-text">{info["insight"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Winning Event Case Study ──
    WINNING_EVENTS = {
        "M2": {
            "ticker": "TSLA", "date": "2024-04-23",
            "title": "The Contrarian Triumph",
            "surprise": "-8.1%", "ret": "+26.7%",
            "tags": ["MISS", "MASSIVE RALLY", "+53.4% swing vs M0"],
            "story": "Tesla was in crisis mode — stock down 43%, Cybertruck reviews devastating, mass layoffs. "
                     "Sentiment at 0.323, the lowest in the entire dataset. Then earnings missed as expected — "
                     "the bad news was fully priced in. Market pivoted to the cheaper EV announcement. "
                     "M2 saw what M0 couldn't: when sentiment is this negative, even a miss becomes a clearing event.",
        },
        "M3": {
            "ticker": "MSFT", "date": "2025-01-29",
            "title": "DeepSeek Week — Network Wins",
            "surprise": "+4.1%", "ret": "-6.6%",
            "tags": ["BEAT", "DROPPED", "sys_conn = 0.535 (top 5%)"],
            "story": "MSFT beat earnings, but system connectedness was at 0.535 — second highest ever. "
                     "The entire Mag7 was running hot with peak AI optimism (sentiment 0.748, highest in dataset). "
                     "Then DeepSeek dropped. M3 recognized this as system-wide hype and shorted. "
                     "M2 predicted UP (wrong). M3's spillover network saw what company-level sentiment couldn't.",
        },
        "XGB": {
            "ticker": "TSLA", "date": "2024-10-23",
            "title": "Regime Threshold — Go Long",
            "surprise": "+24.8%", "ret": "+20.5%",
            "tags": ["BEAT", "RALLIED", "+41.1% edge vs LogReg"],
            "story": "Tesla crushed earnings by 24.8% while sentiment was nearly neutral (0.49) and "
                     "system connectedness was moderate (0.402). LogReg's linear negative coefficient on sentiment "
                     "pushed it to short — wrong. XGBoost's depth-2 tree learned the threshold: "
                     "moderate connectedness + massive beat + non-extreme sentiment = the beat is real, go long.",
        },
    }

    # Only show case study when model + ticker combo matches
    case = WINNING_EVENTS.get(active_model if not show_summary else None)
    is_featured = case and case["ticker"] == ticker

    if case and not is_featured:
        # Model selected but wrong ticker — show a subtle hint
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#71717A;'
            f'padding:8px 0;text-align:center;">'
            f'★ Featured case study available — switch to '
            f'<span style="color:#A855F7;font-weight:600;">{case["ticker"]}</span></div>',
            unsafe_allow_html=True
        )

    if is_featured:
        # ── Star-featured case study ──
        st.markdown('<hr>', unsafe_allow_html=True)

        ret_val = case["ret"]
        is_positive = ret_val.startswith("+")
        ret_color = "#22C55E" if is_positive else "#EF4444"
        ret_bg = "rgba(34,197,94,0.08)" if is_positive else "rgba(239,68,68,0.08)"
        model_color = MODEL_INFO[active_model]["color"]

        st.markdown(
            f'<div style="background:linear-gradient(135deg, rgba(168,85,247,0.08) 0%, rgba(245,158,11,0.05) 100%);'
            f'border:1px solid {model_color}44;border-radius:8px;padding:16px;position:relative;overflow:hidden;">'
            # Star badge
            f'<div style="position:absolute;top:8px;right:10px;font-size:20px;opacity:0.3;">★</div>'
            # Header
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:2px;'
            f'color:{model_color};margin-bottom:6px;">★ FEATURED CASE STUDY</div>'
            f'<div style="font-family:Playfair Display,serif;font-size:18px;color:#FFF;font-weight:700;'
            f'margin-bottom:4px;">{case["title"]}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#A1A1AA;'
            f'margin-bottom:10px;">{case["ticker"]} — {case["date"]}</div>'
            # Metrics row
            f'<div style="display:flex;gap:8px;margin-bottom:12px;">'
            f'<div style="flex:1;background:#0D0D10;border-radius:4px;padding:8px;text-align:center;">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">SURPRISE</div>'
            f'<div style="font-family:Playfair Display,serif;font-size:18px;font-weight:700;'
            f'color:{"#22C55E" if not case["surprise"].startswith("-") else "#EF4444"};">{case["surprise"]}</div></div>'
            f'<div style="flex:1;background:#0D0D10;border-radius:4px;padding:8px;text-align:center;">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">5D RETURN</div>'
            f'<div style="font-family:Playfair Display,serif;font-size:18px;font-weight:700;'
            f'color:{ret_color};">{ret_val}</div></div></div>'
            # Tags
            f'<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;">'
            + ''.join(
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:9px;'
                f'background:rgba(255,255,255,0.04);border:1px solid #27272A;color:#A1A1AA;'
                f'padding:2px 7px;border-radius:3px;">{t}</span>'
                for t in case["tags"]
            )
            + f'</div>'
            # Story
            f'<div style="font-family:Inter,sans-serif;font-size:11.5px;color:#A1A1AA;'
            f'line-height:1.65;border-top:1px solid #27272A;padding-top:10px;">{case["story"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Dynamic Spillover Network (Streamlit slider + static Plotly) ──
    if snapshots:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">SPILLOVER NETWORK</div>', unsafe_allow_html=True)

        # Build select_slider options: "AAPL 2022-10-27" → index
        snap_options = []
        for i, s in enumerate(snapshots):
            parts = s["tag"].split(" ")
            snap_options.append(f"{parts[0]} {parts[1]}")

        selected_snap = st.select_slider(
            "Network Timeline",
            options=snap_options,
            value=snap_options[-1],
            label_visibility="collapsed",
            key="net_timeline",
        )
        snap_idx = snap_options.index(selected_snap)
        snap = snapshots[snap_idx]

        net_fig, sys_conn, conn_color = build_network_dynamic(snap)

        # Header
        parts = snap["tag"].split(" ")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:2px 0 4px;">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#FFF;font-weight:600;">'
            f'{parts[0]}  {parts[1]}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{conn_color};font-weight:600;">'
            f'Conn: {sys_conn:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(net_fig, use_container_width=True, config={"displayModeBar": False})

        # Legend
        st.markdown(
            '<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">'
            '<span style="color:#A855F7;">●</span> Transmitter</span>'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">'
            '<span style="color:#3B82F6;">●</span> Receiver</span>'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">'
            'Size = influence</span>'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#71717A;">'
            'Tight = high connectedness</span>'
            '</div>',
            unsafe_allow_html=True
        )
