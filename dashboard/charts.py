"""
Plotly chart builders for the Spillover Alpha dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Color Palette ─────────────────────────────────────────────────────────

COLORS = {
    "bg": "#0A0A0A",
    "card": "#111114",
    "chart_bg": "#0D0D10",
    "grid": "#1A1A1F",
    "border": "#27272A",
    "text": "#FFFFFF",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    "accent": "#A855F7",
    "green": "#22C55E",
    "red": "#EF4444",
    "yellow": "#F59E0B",
    "blue": "#3B82F6",
}

MODEL_COLORS = {
    "M0": "#6B7280",
    "M1": "#94A3B8",
    "M2": "#3B82F6",
    "M3": "#A855F7",
    "XGB": "#F59E0B",
}

MODEL_NAMES = {
    "M0": "M0 Naive",
    "M1": "M1 Surprise",
    "M2": "M2 Sentiment",
    "M3": "M3 Spillover",
    "XGB": "XGBoost",
}

MAG7_COLORS = {
    "AAPL": "#22C55E",
    "MSFT": "#3B82F6",
    "GOOGL": "#A855F7",
    "AMZN": "#F59E0B",
    "NVDA": "#EF4444",
    "META": "#06B6D4",
    "TSLA": "#F472B6",
}


def build_price_chart(
    prices: pd.DataFrame,
    events_df: pd.DataFrame,
    ticker: str,
    selected_models: list[str],
) -> go.Figure:
    """Build the main interactive stock price chart with earnings + model overlays."""

    fig = go.Figure()

    if ticker == "ALL":
        # ── ALL: normalized overlay of all Mag7 ──
        for tk, color in MAG7_COLORS.items():
            tkp = prices[prices["ticker"] == tk].sort_values("date").copy()
            if tkp.empty:
                continue
            base = tkp.iloc[0]["close"]
            tkp["norm"] = (tkp["close"] / base - 1) * 100
            fig.add_trace(go.Scatter(
                x=tkp["date"], y=tkp["norm"],
                mode="lines", name=tk,
                line=dict(color=color, width=1.5),
                hovertemplate=(
                    "<b>" + tk + "</b><br>"
                    "%{x|%Y-%m-%d}<br>"
                    "Return: %{y:+.1f}%<br>"
                    "<extra></extra>"
                ),
            ))

        fig.update_yaxes(title_text="Normalized Return %", ticksuffix="%")
        fig.update_layout(hovermode="x unified")

        # Common layout
        fig.update_layout(
            plot_bgcolor=COLORS["chart_bg"],
            paper_bgcolor=COLORS["bg"],
            font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=11),
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=dict(showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, showline=False),
            legend=dict(
                orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0,
                font=dict(size=10, color=COLORS["text_muted"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=500,
        )
        return fig


    else:
        # ── Single ticker: candlestick-style with OHLC hover ──
        tkp = prices[prices["ticker"] == tk].sort_values("date").copy() if False else \
              prices[prices["ticker"] == ticker].sort_values("date").copy()

        # Main price line with area fill
        fig.add_trace(go.Scatter(
            x=tkp["date"], y=tkp["close"],
            mode="lines", name=f"{ticker} Price",
            line=dict(color=COLORS["green"], width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(COLORS["green"], 0.05),
            customdata=np.stack([
                tkp["open"].values,
                tkp["high"].values,
                tkp["low"].values,
                tkp["volume"].values,
            ], axis=-1),
            hovertemplate=(
                f"<b>{ticker}</b>  %{{x|%Y-%m-%d}}<br>"
                "────────────<br>"
                "Close: <b>$%{y:.2f}</b><br>"
                "Open: $%{customdata[0]:.2f}<br>"
                "High: $%{customdata[1]:.2f}<br>"
                "Low: $%{customdata[2]:.2f}<br>"
                "Volume: %{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        ))

        # Earnings event markers + model predictions
        tk_events = events_df[
            (events_df["ticker"] == ticker) & events_df["is_test"]
        ].sort_values("earnings_date")

        # Model prediction markers at earnings dates
        for model_key in selected_models:
            pred_col = f"{model_key.lower()}_pred"
            if pred_col not in tk_events.columns:
                continue
            model_ev = tk_events[tk_events[pred_col].notna()].copy()
            if model_ev.empty:
                continue

            model_color = MODEL_COLORS[model_key]

            # Featured events: model+ticker combos that get a star
            FEATURED = {
                ("M2", "TSLA"): "2024-04-23",
                ("M3", "MSFT"): "2025-01-29",
                ("XGB", "TSLA"): "2024-10-23",
            }
            featured_date = FEATURED.get((model_key, ticker))

            # Collect buy/sell/star points for batch traces
            buy_x, buy_y, buy_custom = [], [], []
            sell_x, sell_y, sell_custom = [], [], []
            star_x, star_y, star_custom = [], [], []

            for _, ev in model_ev.iterrows():
                ed = ev["earnings_date"]
                price_at = tkp[tkp["date"] <= ed]
                if price_at.empty:
                    continue
                py = price_at.iloc[-1]["close"]

                pred = int(ev[pred_col])
                actual = int(ev["target"])
                correct = pred == actual
                ret5 = ev["ret_5d"] * 100
                surprise = ev["surprise_pct"]
                check = "✓" if correct else "✗"

                row = [surprise, ret5, correct, check, ev["earnings_date"].strftime("%Y-%m-%d")]

                if featured_date and ed.strftime("%Y-%m-%d") == featured_date:
                    star_x.append(ed); star_y.append(py); star_custom.append(row)
                elif pred == 1:
                    buy_x.append(ed); buy_y.append(py); buy_custom.append(row)
                else:
                    sell_x.append(ed); sell_y.append(py); sell_custom.append(row)

            # Blue solid triangle-up for BUY
            model_label = MODEL_NAMES[model_key]
            if buy_x:
                fig.add_trace(go.Scatter(
                    x=buy_x, y=buy_y,
                    mode="markers",
                    marker=dict(
                        size=15, color="#3B82F6",
                        symbol="triangle-up",
                        line=dict(width=1, color="#93C5FD"),
                    ),
                    customdata=buy_custom,
                    showlegend=False,
                    hovertemplate=(
                        "<b>" + ticker + "</b>  %{customdata[4]}<br>"
                        "────────────<br>"
                        "<b>" + model_label + ": LONG ▲</b>  %{customdata[3]}<br>"
                        "EPS Surprise: %{customdata[0]:+.1f}%<br>"
                        "5d Return: %{customdata[1]:+.1f}%<br>"
                        "Price: $%{y:.2f}"
                        "<extra></extra>"
                    ),
                ))

            # Red solid triangle-down for SELL
            if sell_x:
                fig.add_trace(go.Scatter(
                    x=sell_x, y=sell_y,
                    mode="markers",
                    marker=dict(
                        size=15, color="#EF4444",
                        symbol="triangle-down",
                        line=dict(width=1, color="#FCA5A5"),
                    ),
                    customdata=sell_custom,
                    showlegend=False,
                    hovertemplate=(
                        "<b>" + ticker + "</b>  %{customdata[4]}<br>"
                        "────────────<br>"
                        "<b>" + model_label + ": SHORT ▼</b>  %{customdata[3]}<br>"
                        "EPS Surprise: %{customdata[0]:+.1f}%<br>"
                        "5d Return: %{customdata[1]:+.1f}%<br>"
                        "Price: $%{y:.2f}"
                        "<extra></extra>"
                    ),
                ))

            # Gold star for featured event
            if star_x:
                fig.add_trace(go.Scatter(
                    x=star_x, y=star_y,
                    mode="markers",
                    marker=dict(
                        size=22, color="#F59E0B",
                        symbol="star",
                        line=dict(width=1.5, color="#FDE68A"),
                    ),
                    customdata=star_custom,
                    showlegend=False,
                    hovertemplate=(
                        "<b>★ FEATURED — " + ticker + "</b>  %{customdata[4]}<br>"
                        "────────────<br>"
                        "<b>" + model_label + "'s Biggest Win</b><br>"
                        "EPS Surprise: %{customdata[0]:+.1f}%<br>"
                        "5d Return: %{customdata[1]:+.1f}%<br>"
                        "Price: $%{y:.2f}"
                        "<extra></extra>"
                    ),
                ))

        # Legend entries for active models
        for model_key in selected_models:
            mc = MODEL_COLORS[model_key]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=8, color=mc, symbol="diamond"),
                name=MODEL_NAMES[model_key],
            ))

        fig.update_yaxes(title_text="Price ($)", tickprefix="$")
        fig.update_layout(hovermode="x unified")

    # ── Common layout ──
    fig.update_layout(
        plot_bgcolor=COLORS["chart_bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=11),
        margin=dict(l=60, r=20, t=10, b=40),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False, showline=False,
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0,
            font=dict(size=10, color=COLORS["text_muted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=500,
    )

    return fig


def build_cumulative_return_chart(events_df: pd.DataFrame,
                                  selected_models: list[str],
                                  ticker: str) -> go.Figure:
    """
    Build cumulative return comparison chart.
    Uses the test set matching the highest-priority model selected:
      XGB/M3 → 29-event set (need spillover features)
      M2      → 48-event set (need sentiment features)
      M0/M1   → full 51-event set
    All models in the chart are evaluated on the SAME test set for fair comparison.
    """
    fig = go.Figure()

    if ticker == "ALL":
        test_df = events_df[events_df["is_test"]].copy()
    else:
        test_df = events_df[(events_df["ticker"] == ticker) & events_df["is_test"]].copy()

    test_df = test_df.sort_values("earnings_date")

    # Determine test set scope by highest-priority model
    # Priority: XGB > M3 > M2 > M1/M0
    has_xgb_or_m3 = any(m in selected_models for m in ["XGB", "M3"])
    has_m2 = "M2" in selected_models

    if has_xgb_or_m3:
        # 29-event set: only events with complete spillover features
        scope_df = test_df[test_df["m3_pred"].notna()].copy()
        scope_label = f"{len(scope_df)}-event set (spillover scope)"
    elif has_m2:
        # 48-event set: events with complete sentiment features
        scope_df = test_df[test_df["m2_pred"].notna()].copy()
        scope_label = f"{len(scope_df)}-event set (sentiment scope)"
    elif selected_models:
        # Full test set
        scope_df = test_df.copy()
        scope_label = f"{len(scope_df)}-event set (full scope)"
    else:
        scope_df = test_df.copy()
        scope_label = ""

    for model_key in selected_models:
        pred_col = f"{model_key.lower()}_pred"
        if pred_col not in scope_df.columns:
            continue
        valid = scope_df[scope_df[pred_col].notna()].copy()
        if valid.empty:
            continue

        valid["strat_ret"] = np.where(
            valid[pred_col] == 1, valid["ret_5d"], -valid["ret_5d"]
        )
        valid["cum_ret"] = (1 + valid["strat_ret"]).cumprod() - 1
        final_ret = valid["cum_ret"].iloc[-1] * 100

        fig.add_trace(go.Scatter(
            x=valid["earnings_date"],
            y=valid["cum_ret"] * 100,
            mode="lines+markers",
            name=f"{MODEL_NAMES[model_key]}  ({final_ret:+.0f}%)",
            line=dict(color=MODEL_COLORS[model_key], width=2),
            marker=dict(size=4),
            hovertemplate=(
                "<b>" + MODEL_NAMES[model_key] + "</b><br>"
                "%{x|%Y-%m-%d}<br>"
                "Cumulative: <b>%{y:+.1f}%</b>"
                "<extra></extra>"
            ),
        ))

    # Empty state
    if not selected_models:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Select a model above to compare strategy returns",
            showarrow=False,
            font=dict(size=13, color=COLORS["text_muted"], family="Inter"),
        )

    # Buy & Hold baseline on the same scope
    if not scope_df.empty and selected_models:
        bh = scope_df.copy()
        bh["cum_bh"] = (1 + bh["ret_5d"]).cumprod() - 1
        bh_final = bh["cum_bh"].iloc[-1] * 100
        fig.add_trace(go.Scatter(
            x=bh["earnings_date"], y=bh["cum_bh"] * 100,
            mode="lines",
            name=f"Buy & Hold  ({bh_final:+.0f}%)",
            line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
        ))

    # Scope label annotation
    if scope_label and selected_models:
        fig.add_annotation(
            x=0.01, y=0.98, xref="paper", yref="paper",
            text=scope_label, showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=9, color=COLORS["text_muted"], family="IBM Plex Mono"),
        )

    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1, dash="dash"))

    fig.update_layout(
        plot_bgcolor=COLORS["chart_bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"],
            title_text="Cumulative Return %", zeroline=False,
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        height=240,
        hovermode="x unified",
    )
    return fig


def build_network_graph(snapshots: list, ticker: str,
                        earnings_date: str) -> go.Figure | None:
    """Build a spillover network graph for a specific event."""
    tag = f"{ticker} {earnings_date}"
    snapshot = None
    for s in snapshots:
        if s["tag"] == tag:
            snapshot = s
            break
    if snapshot is None:
        return None

    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = {node["ticker"]: (np.cos(a) * 1.5, np.sin(a) * 1.5)
           for node, a in zip(nodes, angles)}

    fig = go.Figure()

    for edge in edges:
        src_tk = nodes[edge["source"]]["ticker"]
        tgt_tk = nodes[edge["target"]]["ticker"]
        x0, y0 = pos[src_tk]
        x1, y1 = pos[tgt_tk]
        w = edge["weight"]
        if w < 0.05:
            continue
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=hex_to_rgba(COLORS["accent"], 0.27),
                      width=max(0.5, w * 8)),
            showlegend=False, hoverinfo="skip",
        ))

    for node in nodes:
        tk = node["ticker"]
        x, y = pos[tk]
        is_target = tk == ticker
        net = node["net"]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(
                size=24 if is_target else 18,
                color=COLORS["accent"] if is_target else COLORS["blue"],
                line=dict(width=2,
                          color=COLORS["text"] if is_target else COLORS["border"]),
            ),
            text=[tk], textposition="top center",
            textfont=dict(size=10,
                          color=COLORS["text"] if is_target else COLORS["text_secondary"]),
            showlegend=False,
            hovertemplate=(
                f"<b>{tk}</b><br>"
                f"Net Transmitter: {net:+.3f}<br>"
                f"In-degree: {node['in_degree']:.3f}<br>"
                f"Out-degree: {node['out_degree']:.3f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        plot_bgcolor=COLORS["card"],
        paper_bgcolor=COLORS["card"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-2.5, 2.5]),
        yaxis=dict(visible=False, range=[-2.5, 2.5], scaleanchor="x"),
        height=280,
    )
    return fig


def build_network_animated(snapshots: list) -> go.Figure:
    """
    Build a fully client-side animated spillover network using Plotly frames.
    Dragging the slider renders instantly without Streamlit rerun.
    """
    if not snapshots:
        return go.Figure()

    # Precompute layout for all snapshots
    MAG7_ORDER = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    n = 7
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2

    def _snapshot_to_data(snap):
        """Convert one snapshot into trace data dicts."""
        nodes = snap["nodes"]
        edges = snap["edges"]
        sys_conn = snap["system_connectedness"]
        tag = snap["tag"]

        radius = 2.2 - sys_conn * 2.5
        radius = max(0.8, min(2.0, radius))

        pos = {}
        for node in nodes:
            idx = MAG7_ORDER.index(node["ticker"]) if node["ticker"] in MAG7_ORDER else 0
            a = base_angles[idx]
            pos[node["ticker"]] = (np.cos(a) * radius, np.sin(a) * radius)

        out_degrees = [nd["out_degree"] for nd in nodes]
        od_min, od_max = min(out_degrees), max(out_degrees)
        od_range = od_max - od_min if od_max > od_min else 1

        # Edge trace
        edge_x, edge_y = [], []
        for edge in edges:
            src_tk = nodes[edge["source"]]["ticker"]
            tgt_tk = nodes[edge["target"]]["ticker"]
            w = edge["weight"]
            if w < 0.04:
                continue
            x0, y0 = pos[src_tk]
            x1, y1 = pos[tgt_tk]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Node traces
        node_x, node_y, node_size, node_color, node_text, node_hover = [], [], [], [], [], []
        for node in nodes:
            tk = node["ticker"]
            x, y = pos[tk]
            od = node["out_degree"]
            net = node["net"]
            norm_od = (od - od_min) / od_range if od_range > 0 else 0.5
            size = 18 + norm_od * 28

            if net > 0.02:
                color = "#A855F7"
            elif net < -0.02:
                color = "#3B82F6"
            else:
                color = "#71717A"

            node_x.append(x)
            node_y.append(y)
            node_size.append(size)
            node_color.append(color)
            node_text.append(tk)
            node_hover.append(
                f"<b>{tk}</b><br>"
                f"Out: {od:.3f}  In: {node['in_degree']:.3f}<br>"
                f"Net: {net:+.3f}"
            )

        return edge_x, edge_y, node_x, node_y, node_size, node_color, node_text, node_hover, sys_conn, tag

    # Build initial frame from last snapshot
    init = _snapshot_to_data(snapshots[-1])

    fig = go.Figure(
        data=[
            # Trace 0: edges
            go.Scatter(
                x=init[0], y=init[1], mode="lines",
                line=dict(color=hex_to_rgba(COLORS["accent"], 0.2), width=1.5),
                hoverinfo="skip", showlegend=False,
            ),
            # Trace 1: nodes
            go.Scatter(
                x=init[2], y=init[3], mode="markers+text",
                marker=dict(size=init[4], color=init[5],
                            line=dict(width=1.5, color=COLORS["border"])),
                text=init[6], textposition="middle center",
                textfont=dict(size=9, color="#FFF", family="IBM Plex Mono"),
                hovertext=init[7], hoverinfo="text",
                showlegend=False,
            ),
            # Trace 2: connectedness annotation dot (invisible, for title update)
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=0), showlegend=False, hoverinfo="skip",
            ),
        ],
    )

    # Build all frames
    frames = []
    slider_steps = []
    for i, snap in enumerate(snapshots):
        d = _snapshot_to_data(snap)
        sys_conn = d[8]
        tag = d[9]
        parts = tag.split(" ")

        frames.append(go.Frame(
            data=[
                go.Scatter(x=d[0], y=d[1]),
                go.Scatter(x=d[2], y=d[3],
                           marker=dict(size=d[4], color=d[5],
                                       line=dict(width=1.5, color=COLORS["border"])),
                           text=d[6], hovertext=d[7]),
                go.Scatter(),  # placeholder
            ],
            name=str(i),
            layout=go.Layout(
                title=dict(
                    text=(f"<span style='font-size:12px;color:#FFF'>{parts[0]}  {parts[1]}</span>"
                          f"  <span style='font-size:11px;color:"
                          f"{'#EF4444' if sys_conn > 0.5 else '#22C55E' if sys_conn < 0.35 else '#F59E0B'}"
                          f"'>Connectedness: {sys_conn:.3f}</span>"),
                ),
            ),
        ))

        slider_steps.append(dict(
            args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=parts[1][5:] if len(parts) > 1 else str(i),  # show MM-DD
            method="animate",
        ))

    fig.frames = frames

    fig.update_layout(
        plot_bgcolor=COLORS["card"],
        paper_bgcolor=COLORS["card"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=10),
        margin=dict(l=5, r=5, t=30, b=5),
        xaxis=dict(visible=False, range=[-2.5, 2.5]),
        yaxis=dict(visible=False, range=[-2.5, 2.5], scaleanchor="x"),
        height=340,
        title=dict(
            text=f"<span style='font-size:12px;color:#FFF'>{snapshots[-1]['tag']}</span>",
            x=0.01, xanchor="left",
        ),
        sliders=[dict(
            active=len(snapshots) - 1,
            currentvalue=dict(visible=False),
            pad=dict(t=0, b=10),
            steps=slider_steps,
            bgcolor=COLORS["card"],
            activebgcolor=COLORS["accent"],
            bordercolor=COLORS["border"],
            font=dict(size=8, color=COLORS["text_muted"]),
            len=1.0, x=0, xanchor="left",
        )],
        updatemenus=[],  # no play button
    )

    return fig


def build_network_dynamic(snapshot: dict) -> go.Figure:
    """
    Build a dynamic spillover network graph from a snapshot.
    Node size = out_degree (intensity of influence).
    Edge width = spillover weight.
    Layout radius contracts with higher system_connectedness (tighter cluster).
    """
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    sys_conn = snapshot["system_connectedness"]

    # Radius inversely proportional to connectedness: high conn → tight cluster
    # sys_conn range ~0.30-0.56, map to radius ~2.0-1.0
    radius = 2.2 - sys_conn * 2.5
    radius = max(0.8, min(2.0, radius))

    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = {node["ticker"]: (np.cos(a) * radius, np.sin(a) * radius)
           for node, a in zip(nodes, angles)}

    # Node size: map out_degree to pixel size
    # out_degree range ~0.2-0.65 → size 14-40
    out_degrees = [nd["out_degree"] for nd in nodes]
    od_min, od_max = min(out_degrees), max(out_degrees)
    od_range = od_max - od_min if od_max > od_min else 1

    fig = go.Figure()

    # Edges
    for edge in edges:
        src_tk = nodes[edge["source"]]["ticker"]
        tgt_tk = nodes[edge["target"]]["ticker"]
        x0, y0 = pos[src_tk]
        x1, y1 = pos[tgt_tk]
        w = edge["weight"]
        if w < 0.04:
            continue
        # Edge opacity scales with weight
        alpha = min(0.5, w * 2)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=hex_to_rgba(COLORS["accent"], alpha),
                      width=max(0.5, w * 10)),
            showlegend=False, hoverinfo="skip",
        ))

    # Nodes
    for node in nodes:
        tk = node["ticker"]
        x, y = pos[tk]
        od = node["out_degree"]
        net = node["net"]

        # Size by out_degree
        norm_od = (od - od_min) / od_range if od_range > 0 else 0.5
        size = 16 + norm_od * 26

        # Color: transmitters (net > 0) = accent, receivers (net < 0) = blue
        if net > 0.02:
            color = COLORS["accent"]
            border = "#D8B4FE"
        elif net < -0.02:
            color = COLORS["blue"]
            border = "#93C5FD"
        else:
            color = COLORS["text_muted"]
            border = COLORS["border"]

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(
                size=size,
                color=hex_to_rgba(color, 0.7),
                line=dict(width=2, color=border),
            ),
            text=[tk],
            textposition="middle center" if size > 26 else "top center",
            textfont=dict(
                size=9 if size < 26 else 10,
                color=COLORS["text"],
                family="IBM Plex Mono",
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{tk}</b><br>"
                f"Out-degree: {od:.3f}<br>"
                f"In-degree: {node['in_degree']:.3f}<br>"
                f"Net: {net:+.3f}  ({'transmitter' if net > 0 else 'receiver'})"
                "<extra></extra>"
            ),
        ))

    # Sys conn label
    conn_color = COLORS["red"] if sys_conn > 0.5 else COLORS["green"] if sys_conn < 0.35 else COLORS["yellow"]

    fig.update_layout(
        plot_bgcolor=COLORS["card"],
        paper_bgcolor=COLORS["card"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=10),
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False, range=[-2.5, 2.5]),
        yaxis=dict(visible=False, range=[-2.5, 2.5], scaleanchor="x"),
        height=300,
    )
    return fig, sys_conn, conn_color


# ── Progressive Story Chart (ALL view) ────────────────────────────────────

PROGRESSIVE_MODELS = [
    ("M0", "m0_pred", "#6B7280", "M0 Naive",      "Beat = Up, Miss = Down", 1),
    ("M1", "m1_pred", "#94A3B8", "M1 Surprise %",  "+ magnitude of beat/miss", 1),
    ("M2", "m2_pred", "#3B82F6", "M2 + Sentiment", "+ pre-earnings news sentiment", 2),
    ("M3", "m3_pred", "#A855F7", "M3 + Spillover",  "+ cross-company DY network", 2.5),
    ("XGB","xgb_pred","#F59E0B", "XGBoost",        "+ non-linear regime thresholds", 2.5),
]


def _build_progressive_chart(events_df: pd.DataFrame) -> go.Figure:
    """
    Build the ALL-view progressive strategy comparison.
    Shows cumulative return curves for M0→M1→M2→M3→XGB telling the narrative.
    """
    fig = go.Figure()
    test_df = events_df[events_df["is_test"]].sort_values("earnings_date").copy()

    # Zero line
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1, dash="dash"))

    final_returns = {}

    for key, pred_col, color, label, _, width in PROGRESSIVE_MODELS:
        valid = test_df[test_df[pred_col].notna()].copy()
        if valid.empty:
            continue

        valid["strat_ret"] = np.where(
            valid[pred_col] == 1, valid["ret_5d"], -valid["ret_5d"]
        )
        valid["cum_ret"] = (1 + valid["strat_ret"]).cumprod() - 1
        final_ret = valid["cum_ret"].iloc[-1] * 100
        final_returns[key] = final_ret
        n_events = len(valid)

        fig.add_trace(go.Scatter(
            x=valid["earnings_date"],
            y=valid["cum_ret"] * 100,
            mode="lines",
            name=f"{label}  ({final_ret:+.0f}%)",
            line=dict(color=color, width=width),
            hovertemplate=(
                "<b>" + label + "</b><br>"
                "%{x|%Y-%m-%d}<br>"
                "Cumulative: <b>%{y:+.1f}%</b>"
                "<extra></extra>"
            ),
        ))

        # End-of-line label annotation
        fig.add_annotation(
            x=valid["earnings_date"].iloc[-1],
            y=final_ret,
            text=f"<b>{final_ret:+.0f}%</b>",
            showarrow=False,
            xanchor="left", xshift=8,
            font=dict(size=12, color=color, family="IBM Plex Mono"),
        )

    # Buy & Hold baseline
    bh = test_df.copy()
    bh["cum_bh"] = (1 + bh["ret_5d"]).cumprod() - 1
    bh_final = bh["cum_bh"].iloc[-1] * 100
    fig.add_trace(go.Scatter(
        x=bh["earnings_date"], y=bh["cum_bh"] * 100,
        mode="lines", name=f"Buy & Hold  ({bh_final:+.0f}%)",
        line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
    ))

    # Key event annotations
    _add_event_annotations(fig, test_df)

    fig.update_layout(
        plot_bgcolor=COLORS["chart_bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS["text_muted"], size=11),
        margin=dict(l=60, r=80, t=30, b=50),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            title_text="Earnings Events (time-ordered)",
            title_font=dict(size=10, color=COLORS["text_muted"]),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=COLORS["grid"],
            zeroline=False, showline=False,
            title_text="Cumulative Return %", ticksuffix="%",
        ),
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01,
            font=dict(size=11), bgcolor=hex_to_rgba(COLORS["card"], 0.85),
            bordercolor=COLORS["border"], borderwidth=1,
        ),
        height=560,
        hovermode="x unified",
    )

    return fig


def _add_event_annotations(fig: go.Figure, test_df: pd.DataFrame):
    """Add key narrative event markers to the progressive chart."""
    key_events = {
        "TSLA 2024-04-23": ("TSLA miss → +26.7% rally", COLORS["green"]),
        "MSFT 2025-01-29": ("DeepSeek shock week", COLORS["red"]),
        "NVDA 2024-11-20": ("Peak connectedness", COLORS["red"]),
        "NVDA 2024-02-21": ("AI supercycle beat", COLORS["green"]),
    }
    for _, ev in test_df.iterrows():
        tag = f"{ev['ticker']} {ev['earnings_date'].strftime('%Y-%m-%d')}"
        if tag in key_events:
            label, color = key_events[tag]
            fig.add_vline(
                x=ev["earnings_date"],
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.3,
            )
            fig.add_annotation(
                x=ev["earnings_date"], y=1.0, yref="paper",
                text=f"<b>{ev['ticker']}</b><br><span style='font-size:9px'>{label}</span>",
                showarrow=False, yanchor="top", yshift=-5,
                font=dict(size=10, color=color, family="IBM Plex Mono"),
                bgcolor=hex_to_rgba(color, 0.08),
                bordercolor=color, borderwidth=1, borderpad=3,
            )
