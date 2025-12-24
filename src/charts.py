from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import COR_PRINCIPAL, COR_SECUNDARIA, GRID_Y, PANEL_W, PANEL_H


def _layout_base(fig: go.Figure, titulo: str, y_titulo: str) -> go.Figure:
    fig.update_layout(
        title=titulo,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
        width=PANEL_W,
        height=PANEL_H,
    )
    fig.update_yaxes(
        title=y_titulo,
        ticksuffix="%",
        tickformat=".0f",
        showgrid=True,
        gridcolor=GRID_Y,
        zeroline=False,
        showline=False,
    )
    fig.update_xaxes(
        tickangle=90,
        showgrid=False,
        zeroline=False,
    )
    return fig


def grafico_yoy_cpi_vs_trim(df_yoy: pd.DataFrame) -> go.Figure:
    ticks = pd.date_range("1996-01-01", df_yoy["REF_DATE"].max(), freq="YS")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_yoy["REF_DATE"], y=df_yoy["cpi_trim_yoy_pct"],
        mode="lines", name="CPI-Trim YoY",
        line=dict(color=COR_PRINCIPAL, width=2.7),
    ))
    fig.add_trace(go.Scatter(
        x=df_yoy["REF_DATE"], y=df_yoy["cpi_yoy_pct"],
        mode="lines", name="CPI YoY",
        line=dict(color=COR_SECUNDARIA, width=2.2),
    ))
    for y in [1, 2, 3]:
        fig.add_hline(y=y, line_width=1, line_dash="dot", opacity=0.35)

    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[d.strftime("%b-%y") for d in ticks],
    )
    return _layout_base(fig, "Canada Inflation — CPI vs CPI-Trim (YoY)", "Percent")


def grafico_curto_mom_saar_mm3(df_curto: pd.DataFrame) -> go.Figure:
    ticks = pd.date_range("2011-01-01", df_curto["REF_DATE"].max(), freq="6MS")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curto["REF_DATE"], y=df_curto["cpi_trim_mom_saar_pct"],
        mode="lines", name="MoM SAAR",
        line=dict(color=COR_SECUNDARIA, width=2.2),
        opacity=0.65,
    ))
    fig.add_trace(go.Scatter(
        x=df_curto["REF_DATE"], y=df_curto["cpi_trim_mm3_saar_pct"],
        mode="lines", name="MM3 (annualized)",
        line=dict(color=COR_PRINCIPAL, width=2.7),
    ))
    for y in [2, 3]:
        fig.add_hline(y=y, line_width=1, line_dash="dot", opacity=0.35)

    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[d.strftime("%b-%y") for d in ticks],
    )
    return _layout_base(fig, "CPI-Trim — MoM SAAR vs MM3 (since 2011)", "Percent (annualized)")


def grafico_componentes_mom(det_inc: pd.DataFrame, top_n: int = 30) -> go.Figure:
    mom = det_inc[["componente", "mom_pct"]].sort_values("mom_pct", ascending=False).copy()
    mom_pos = mom[mom["mom_pct"] >= 0].head(top_n // 2)
    mom_neg = mom[mom["mom_pct"] < 0].tail(top_n // 2)
    mom_plot = pd.concat([mom_pos, mom_neg], ignore_index=True).sort_values("mom_pct")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mom_plot["mom_pct"],
        y=mom_plot["componente"],
        orientation="h",
        marker_color=COR_SECUNDARIA,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.35)")

    fig.update_layout(
        title="Componentes incluídos — MoM SA (m/m)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=30, t=70, b=60),
        width=PANEL_W,
        height=PANEL_H,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False, title="Percent (m/m)")
    fig.update_yaxes(showgrid=False, automargin=True, title="")
    return fig


def grafico_componentes_contrib(det_inc: pd.DataFrame, top_n: int = 30) -> go.Figure:
    contrib = det_inc[["componente", "contrib_bps"]].copy()
    contrib["abs"] = contrib["contrib_bps"].abs()
    contrib_plot = contrib.sort_values("abs", ascending=False).head(top_n).sort_values("contrib_bps")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contrib_plot["contrib_bps"],
        y=contrib_plot["componente"],
        orientation="h",
        marker_color=COR_PRINCIPAL,
        hovertemplate="%{y}<br>%{x:.1f} bps<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.35)")

    fig.update_layout(
        title="Componentes incluídos — Contribuição ao CPI-Trim (bps)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=220, r=30, t=70, b=60),
        width=PANEL_W,
        height=PANEL_H,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False, title="bps")
    fig.update_yaxes(showgrid=False, automargin=True, title="")
    return fig
