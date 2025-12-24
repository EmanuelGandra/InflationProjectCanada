"""
cpi_trim_canada.py

Projeto (estudo): replicar uma versão do CPI-Trim do Canadá (BoC) usando dados do Statistics Canada.

O que este script faz
1) Baixa CPI (níveis) e pesos do basket via pacote `stats_can`
2) Filtra Canadá + lista oficial de 55 componentes
3) Ajuste sazonal por item (STL)  [obs: não faz ajuste de impostos] -> Instalar o X13-ARIMA
4) Calcula MoM (m/m) do índice dessazonalizado
5) Calcula CPI-Trim (m/m) com trimming de 20%/20% por pesos
6) Calcula métricas derivadas:
   - YoY (12m composto)
   - MoM SAAR (anualizado por composição)
   - MM3 anualizado (média móvel 3m do MoM e depois anualiza por composição)
7) Gera gráficos em Plotly e salva um relatório HTML (e tenta PDF, se tiver kaleido)

Estilo: código organizado e comentado “o suficiente”, com nomes em português.

Requisitos:
- pandas, numpy, statsmodels, plotly, stats_can
- (opcional p/ PDF) kaleido

Como rodar:
python cpi_trim_canada.py --saida_dir ../output --cache_dir ../data

Dica:
Se o PDF não sair, rode:
pip install -U kaleido
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
import io

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from statsmodels.tsa.seasonal import STL

import stats_can
from stats_can import sc


# =========================================================
# CONFIG: warnings (limpar o console)
# =========================================================
def configurar_warnings() -> None:
    """
    - stats_can hoje dispara FutureWarning interno no to_datetime(errors='ignore')
    - também podem aparecer RuntimeWarnings pontuais em divisões (a gente trata, mas filtro é ok)
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*errors='ignore' is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*DataFrameGroupBy\.apply operated on the grouping columns.*",
        category=FutureWarning,
    )
    # Se você preferir VER RuntimeWarnings, pode comentar isso:
    warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# CONFIG VISUAL (paleta e layout)
# =========================================================
COR_PRINCIPAL = "#c00000"   # destaque
COR_SECUNDARIA = "#5b89c1"  # secundária
GRID_Y = "rgba(0,0,0,0.10)"

# tamanho "padrão" de cada painel do dashboard
PANEL_W = 720
PANEL_H = 420

from pathlib import Path

# Raiz do projeto = pasta pai de /src (onde este arquivo está)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "output"

# =========================================================
# 55 COMPONENTES (Table A1)
# =========================================================
COMPONENTES_55: List[str] = [
    "Meat",
    "Fish, seafood and other marine products",
    "Dairy products and eggs",
    "Bakery and cereal products (excluding baby food)",
    "Fruit, fruit preparations and nuts",
    "Vegetables and vegetable preparations",
    "Other food products and non-alcoholic beverages",
    "Food purchased from restaurants",
    "Rented accommodation",
    "Mortgage interest cost",
    "Homeowners' replacement cost",
    "Property taxes and other special charges",
    "Homeowners' home and mortgage insurance",
    "Homeowners' maintenance and repairs",
    "Other owned accommodation expenses",
    "Electricity",
    "Water",
    "Natural gas",
    "Fuel oil and other fuels",
    "Communications",
    "Child care and housekeeping services",
    "Household cleaning products",
    "Paper, plastic and aluminum foil supplies",
    "Other household goods and services",
    "Furniture",
    "Household textiles",
    "Household equipment",
    "Services related to household furnishings and equipment",
    "Clothing",
    "Footwear",
    "Clothing accessories, watches and jewellery",
    "Clothing material, notions and services",
    "Purchase of passenger vehicles",
    "Leasing of passenger vehicles",
    "Rental of passenger vehicles",
    "Gasoline",
    "Passenger vehicle parts, maintenance and repairs",
    "Other passenger vehicle operating expenses",
    "Local and commuter transportation",
    "Inter-city transportation",
    "Health care goods",
    "Health care services",
    "Personal care supplies and equipment",
    "Personal care services",
    "Recreational equipment and services (excluding recreational vehicles)",
    "Purchase of recreational vehicles and outboard motors",
    "Operation of recreational vehicles",
    "Home entertainment equipment, parts and services",
    "Travel services",
    "Other cultural and recreational services",
    "Education",
    "Reading material (excluding textbooks)",
    "Alcoholic beverages served in licensed establishments",
    "Alcoholic beverages purchased from stores",
    "Tobacco products and smokers' supplies",
]


# =========================================================
# HELPERS: dados
# =========================================================
def carregar_tabelas_statcan(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa (ou usa cache) das tabelas do StatCan via stats_can.
    - CPI:   18-10-0004-13
    - Pesos: 18-10-0007-01
    """
    pid_cpi = "18-10-0004-13"
    pid_pesos = "18-10-0007-01"

    df_cpi = sc.zip_table_to_dataframe(pid_cpi, path=cache_dir)
    df_pesos = sc.zip_table_to_dataframe(pid_pesos, path=cache_dir)
    return df_cpi, df_pesos


def filtrar_canada(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["GEO"].astype(str).str.lower().eq("canada")].copy()


def selecionar_componentes_55(df_can: pd.DataFrame, col_prod: str) -> pd.DataFrame:
    return df_can.query(f"`{col_prod}` in @COMPONENTES_55").copy()


def escolher_pesos_55(
    df_pesos_can: pd.DataFrame,
    col_prod: str,
    data_ref: pd.Timestamp,
    price_period: str = "Weight at basket link month prices",
    geo_dist: str = "Distribution to selected geographies",
) -> pd.DataFrame:
    """
    Seleciona a combinação correta de pesos para ficar com 55 linhas.
    Depois normaliza para somar 1.0
    """
    df_basket = df_pesos_can.query("REF_DATE == @data_ref").copy()

    df_55 = (
        df_basket
        .query("`Price period of weight` == @price_period")
        .query("`Geographic distribution of weight` == @geo_dist")
        .query(f"`{col_prod}` in @COMPONENTES_55")
        .copy()
    )

    df_55["peso"] = df_55["VALUE"].astype(float)
    soma = df_55["peso"].sum()
    if soma == 0 or np.isnan(soma):
        raise RuntimeError("Soma dos pesos veio zero/NaN. Checar filtros de pesos.")
    df_55["peso"] = df_55["peso"] / soma
    return df_55


# =========================================================
# AJUSTE SAZONAL
# =========================================================
def ajuste_sazonal_stl(serie: pd.Series, periodo: int = 12) -> pd.Series:
    """
    Dessazonaliza com STL (robusto).
    Retorna SA = original - sazonal
    """
    serie = serie.dropna()
    if len(serie) < 2 * periodo:
        return serie

    stl = STL(serie, period=periodo, robust=True)
    res = stl.fit()
    return serie - res.seasonal


def dessazonalizar_componentes(df_cpi_55: pd.DataFrame, col_prod: str) -> pd.DataFrame:
    """
    Dessazonaliza item a item e devolve:
    REF_DATE | Products... | indice_sa
    """
    df_cpi_55 = df_cpi_55.sort_values([col_prod, "REF_DATE"]).copy()
    lista = []

    for prod in COMPONENTES_55:
        df_prod = df_cpi_55[df_cpi_55[col_prod] == prod].copy()
        serie = (
            df_prod
            .set_index("REF_DATE")["VALUE"]
            .astype(float)
            .sort_index()
        )

        serie_sa = ajuste_sazonal_stl(serie, periodo=12)
        df_sa = serie_sa.rename("indice_sa").reset_index()
        df_sa[col_prod] = prod
        lista.append(df_sa)

    df_sa_final = pd.concat(lista, ignore_index=True)
    df_sa_final = df_sa_final.sort_values([col_prod, "REF_DATE"]).copy()
    return df_sa_final


# =========================================================
# CPI-TRIM (núcleo)
# =========================================================
def calcular_mom(df_sa: pd.DataFrame, col_prod: str) -> pd.DataFrame:
    df = df_sa.sort_values([col_prod, "REF_DATE"]).copy()
    df["mom"] = df.groupby(col_prod)["indice_sa"].pct_change()
    return df


def cpi_trim_mes(df_mes: pd.DataFrame, lim_inf: float = 0.20, lim_sup: float = 0.80) -> float:
    """
    CPI-Trim do mês (m/m) usando peso efetivo nos 60% centrais.
    """
    df = (
        df_mes
        .dropna(subset=["mom", "peso"])
        .sort_values("mom")
        .copy()
    )
    if df.empty:
        return np.nan

    df["peso_acum"] = df["peso"].cumsum()
    df["peso_efetivo"] = 0.0

    for i, row in df.iterrows():
        w_prev = row["peso_acum"] - row["peso"]
        w_curr = row["peso_acum"]

        if w_curr <= lim_inf:
            continue
        if w_prev >= lim_sup:
            continue

        if (w_prev >= lim_inf) and (w_curr <= lim_sup):
            df.at[i, "peso_efetivo"] = row["peso"]
            continue

        if (w_prev < lim_inf) and (w_curr > lim_inf):
            df.at[i, "peso_efetivo"] = w_curr - lim_inf
            continue

        if (w_prev < lim_sup) and (w_curr > lim_sup):
            df.at[i, "peso_efetivo"] = lim_sup - w_prev
            continue

    denom = df["peso_efetivo"].sum()
    if denom <= 0 or np.isnan(denom):
        # evita RuntimeWarning e te diz que “não deu para calcular naquele mês”
        return np.nan

    trim_mom = (df["mom"] * df["peso_efetivo"]).sum() / denom
    return float(trim_mom)


def detalhar_trim_mes(df_mes: pd.DataFrame, col_prod: str, lim_inf=0.20, lim_sup=0.80) -> pd.DataFrame:
    """
    Detalhe por componente no mês:
    - included (peso_efetivo > 0)
    - share (peso_efetivo / soma_peso_efetivo)
    - mom_pct
    - contrib_bps
    """
    df = (
        df_mes
        .dropna(subset=["mom", "peso"])
        .sort_values("mom")
        .copy()
    )
    if df.empty:
        return df

    df["peso_acum"] = df["peso"].cumsum()
    df["peso_efetivo"] = 0.0

    for i, row in df.iterrows():
        w_prev = row["peso_acum"] - row["peso"]
        w_curr = row["peso_acum"]

        if w_curr <= lim_inf:
            continue
        if w_prev >= lim_sup:
            continue

        if (w_prev >= lim_inf) and (w_curr <= lim_sup):
            df.at[i, "peso_efetivo"] = row["peso"]
            continue

        if (w_prev < lim_inf) and (w_curr > lim_inf):
            df.at[i, "peso_efetivo"] = w_curr - lim_inf
            continue

        if (w_prev < lim_sup) and (w_curr > lim_sup):
            df.at[i, "peso_efetivo"] = lim_sup - w_prev
            continue

    df["included"] = df["peso_efetivo"] > 0
    soma_eff = df.loc[df["included"], "peso_efetivo"].sum()
    df["share"] = np.where(df["included"] & (soma_eff > 0), df["peso_efetivo"] / soma_eff, 0.0)

    df["mom_pct"] = df["mom"] * 100
    df["contrib_bps"] = (df["mom"] * df["share"]) * 10000
    df = df.rename(columns={col_prod: "componente"})
    return df


# =========================================================
# MÉTRICAS (YoY, SAAR, MM3 capitalizando)
# =========================================================
def calcular_metricas(serie: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DF com REF_DATE e cpi_trim_mom (decimal).
    """
    s = serie.sort_values("REF_DATE").copy()

    s["cpi_trim_mom_saar"] = (1 + s["cpi_trim_mom"])**12 - 1
    s["cpi_trim_mm3_mom"] = s["cpi_trim_mom"].rolling(3).mean()
    s["cpi_trim_mm3_saar"] = (1 + s["cpi_trim_mm3_mom"])**12 - 1

    s["cpi_trim_yoy"] = (
        (1 + s["cpi_trim_mom"])
        .rolling(12)
        .apply(lambda x: float(np.prod(x) - 1), raw=False)
    )

    s["cpi_trim_mom_pct"] = 100 * s["cpi_trim_mom"]
    s["cpi_trim_mom_saar_pct"] = 100 * s["cpi_trim_mom_saar"]
    s["cpi_trim_mm3_saar_pct"] = 100 * s["cpi_trim_mm3_saar"]
    s["cpi_trim_yoy_pct"] = 100 * s["cpi_trim_yoy"]

    return s


# =========================================================
# GRÁFICOS (Plotly)
# =========================================================
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
    """
    Barras horizontais: MoM SA (%) – top positivos e negativos.
    """
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
    """
    Barras horizontais: contribuição (bps) – top por contribuição absoluta.
    """
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


# =========================================================
# DASHBOARD HTML (2x2) + PDF único
# =========================================================
def salvar_dashboard_html(
    fig_yoy: go.Figure,
    fig_curto: go.Figure,
    fig_mom: go.Figure,
    fig_contrib: go.Figure,
    out_html: Path,
    subtitulo: str,
) -> None:
    """
    Monta um HTML com grid 2x2 e só inclui o plotly.js uma vez.
    """
    # plotly.js vem aqui (uma vez)
    html_plotlyjs = fig_yoy.to_html(full_html=False, include_plotlyjs="cdn")

    # depois, o resto sem JS
    bloco_yoy = fig_yoy.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    bloco_curto = fig_curto.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    bloco_mom = fig_mom.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    bloco_contrib = fig_contrib.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"

    html = f"""
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Dashboard — CPI-Trim Canadá</title>
    <script src="{plotly_cdn}"></script>
    <style>
      :root {{
        --vermelho: {COR_PRINCIPAL};
        --azul: {COR_SECUNDARIA};
        --texto: #222;
        --muted: #666;
        --bg: #fff;
        --borda: rgba(0,0,0,0.10);
      }}
      body {{
        margin: 0;
        background: var(--bg);
        font-family: Arial, sans-serif;
        color: var(--texto);
      }}
      .topo {{
        padding: 22px 28px 14px 28px;
        border-bottom: 3px solid var(--vermelho);
      }}
      .topo h1 {{
        margin: 0;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.2px;
      }}
      .topo p {{
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 13px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        padding: 18px 18px 26px 18px;
      }}
      .card {{
        border: 1px solid var(--borda);
        border-radius: 10px;
        padding: 10px 10px 6px 10px;
        background: #fff;
      }}
      .card .titulo {{
        font-size: 14px;
        font-weight: 700;
        margin: 6px 8px 8px 8px;
      }}
      /* força o plotly respeitar o card */
      .card .plotly-graph-div {{
        width: 100% !important;
      }}
      @media print {{
        body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        .grid {{ gap: 12px; padding: 12px; }}
        .card {{ border: 1px solid #ddd; }}
      }}
    </style>
  </head>
  <body>
    <div class="topo">
      <h1>Dashboard — CPI-Trim Canadá</h1>
      <p>{subtitulo}</p>
    </div>

    <div class="grid">
      <div class="card">
        <div class="titulo" style="color: var(--vermelho);">YoY — CPI vs CPI-Trim</div>
        {bloco_yoy}
      </div>

      <div class="card">
        <div class="titulo" style="color: var(--vermelho);">Curto prazo — MM3 vs MoM SAAR</div>
        {bloco_curto}
      </div>

      <div class="card">
        <div class="titulo" style="color: var(--azul);">Componentes (MoM SA)</div>
        {bloco_mom}
      </div>

      <div class="card">
        <div class="titulo" style="color: var(--azul);">Componentes (contribuição em bps)</div>
        {bloco_contrib}
      </div>
    </div>
  </body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")

def exportar_dashboard_pdf(
    fig_yoy: go.Figure,
    fig_curto: go.Figure,
    fig_mom: go.Figure,
    fig_contrib: go.Figure,
    out_pdf: Path,
) -> None:
    """
    Gera UM PDF único (1 página) com layout 2x2.
    Implementação: renderiza cada figura em PNG (kaleido) e monta o PDF (reportlab).
    """
    try:
        # plotly usa kaleido internamente; importar garante que está instalado
        import kaleido  # noqa: F401
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.utils import ImageReader
    except Exception:
        print("[Aviso] PDF do dashboard não gerado. Instale: pip install -U kaleido reportlab")
        return

    # renderiza imagens (alta resolução)
    scale = 2.0
    png1 = fig_yoy.to_image(format="png", scale=scale)
    png2 = fig_curto.to_image(format="png", scale=scale)
    png3 = fig_mom.to_image(format="png", scale=scale)
    png4 = fig_contrib.to_image(format="png", scale=scale)

    # >>>>>> AQUI é o ajuste principal: bytes -> BytesIO <<<<<<
    img1 = ImageReader(io.BytesIO(png1))
    img2 = ImageReader(io.BytesIO(png2))
    img3 = ImageReader(io.BytesIO(png3))
    img4 = ImageReader(io.BytesIO(png4))

    # PDF landscape A4 para caber 2x2
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))

    margem = 18
    gap = 10

    util_w = page_w - 2 * margem
    util_h = page_h - 2 * margem

    cell_w = (util_w - gap) / 2
    cell_h = (util_h - gap) / 2

    x1 = margem
    x2 = margem + cell_w + gap
    y2 = margem + cell_h + gap
    y1 = margem

    # desenhar
    c.drawImage(img1, x1, y2, width=cell_w, height=cell_h, preserveAspectRatio=True, anchor="c")
    c.drawImage(img2, x2, y2, width=cell_w, height=cell_h, preserveAspectRatio=True, anchor="c")
    c.drawImage(img3, x1, y1, width=cell_w, height=cell_h, preserveAspectRatio=True, anchor="c")
    c.drawImage(img4, x2, y1, width=cell_w, height=cell_h, preserveAspectRatio=True, anchor="c")

    c.showPage()
    c.save()
    print(f"[OK] PDF do dashboard salvo em: {out_pdf.resolve()}")


def exportar_html_para_pdf(out_html: Path, out_pdf: Path) -> bool:
    """
    Converte o HTML gerado em PDF, mantendo o layout (print-to-pdf).
    Requer Playwright:
      pip install playwright
      python -m playwright install chromium
    Retorna True se gerou, False se não conseguiu (para fallback).
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})

            # IMPORTANTÍSSIMO: usar file:// para carregar recursos locais
            page.goto(out_html.resolve().as_uri(), wait_until="networkidle")

            # imprime em landscape para caber 2x2 (como no HTML)
            page.pdf(
                path=str(out_pdf),
                format="A4",
                landscape=True,
                print_background=True,
                margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
            )
            browser.close()
        print(f"[OK] PDF (via HTML print) salvo em: {out_pdf.resolve()}")
        return True
    except Exception as e:
        print(f"[Aviso] Falha ao gerar PDF via HTML/Playwright: {e}")
        return False
        
# =========================================================
# MAIN
# =========================================================
def main() -> int:
    configurar_warnings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--saida_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Pasta de saída (html/pdf)")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Pasta de cache dos zips do StatCan")
    parser.add_argument("--basket_ref", type=str, default=None, help="Data de referência dos pesos (YYYY-MM-01). Default: última disponível.")
    parser.add_argument("--mes_ref", type=str, default=None, help="Mês dos gráficos de componentes (YYYY-MM-01). Default: último mês do CPI-trim.")
    parser.add_argument("--gerar_pdf", action="store_true", help="Se setado, gera um PDF único do dashboard (kaleido+reportlab)")
    args = parser.parse_args()

    saida_dir = Path(args.saida_dir)
    cache_dir = Path(args.cache_dir)
    saida_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("stats_can path:", stats_can.__file__)
    print("stats_can versão:", getattr(stats_can, "__version__", "desconhecida"))
    print("tem zip_table_to_dataframe?", hasattr(sc, "zip_table_to_dataframe"))

    # 1) baixar dados
    df_cpi_bruto, df_pesos_bruto = carregar_tabelas_statcan(cache_dir)

    # 2) filtrar Canadá
    df_cpi_can = filtrar_canada(df_cpi_bruto)
    df_pesos_can = filtrar_canada(df_pesos_bruto)

    print("CPI Canada shape:", df_cpi_can.shape)
    print("CPI datas min/max:", df_cpi_can["REF_DATE"].min(), df_cpi_can["REF_DATE"].max())
    print("Pesos Canada shape:", df_pesos_can.shape)
    print("Pesos datas min/max:", df_pesos_can["REF_DATE"].min(), df_pesos_can["REF_DATE"].max())

    col_prod = "Products and product groups"

    # 3) 55 componentes
    df_cpi_55 = selecionar_componentes_55(df_cpi_can, col_prod)
    print("CPI 55 shape:", df_cpi_55.shape)
    print("Componentes:", df_cpi_55[col_prod].nunique())

    # 4) data de pesos (basket_ref)
    if args.basket_ref is None:
        data_basket_ref = pd.to_datetime(df_pesos_can["REF_DATE"].max())
    else:
        data_basket_ref = pd.to_datetime(args.basket_ref)

    df_pesos_55 = escolher_pesos_55(df_pesos_can, col_prod, data_basket_ref)
    print("Pesos 55 shape:", df_pesos_55.shape)
    print("Componentes:", df_pesos_55[col_prod].nunique())
    print("Soma dos pesos (normalizado):", df_pesos_55["peso"].sum())

    # 5) dessazonalização
    df_sa = dessazonalizar_componentes(df_cpi_55, col_prod)
    print("Shape CPI SA:", df_sa.shape)

    # 6) MoM
    df_sa_mom = calcular_mom(df_sa, col_prod)

    # 7) merge (MoM + peso)
    df_base = df_sa_mom.merge(
        df_pesos_55[[col_prod, "peso"]],
        on=col_prod,
        how="left",
    )
    if df_base["peso"].isna().any():
        raise RuntimeError("Encontrou peso faltando após merge. Checar mapeamento dos 55 componentes.")

    # 8) série CPI-trim (m/m) — seleciona colunas antes do apply (evita warning futuro)
    df_base_small = df_base[["REF_DATE", "mom", "peso"]].copy()
    serie_trim = (
        df_base_small
        .groupby("REF_DATE", sort=True, group_keys=False)
        .apply(lambda g: cpi_trim_mes(g, lim_inf=0.20, lim_sup=0.80))
        .rename("cpi_trim_mom")
        .reset_index()
        .sort_values("REF_DATE")
    )

    # 9) métricas
    serie_trim = calcular_metricas(serie_trim)

    # 10) CPI headline YoY (All-items)
    df_all = (
        df_cpi_can
        .query("`Products and product groups` == 'All-items'")
        .sort_values("REF_DATE")
        .copy()
    )
    df_all["cpi_yoy"] = df_all["VALUE"].astype(float).pct_change(12)
    serie_cpi_yoy = (
        df_all[["REF_DATE", "cpi_yoy"]]
        .dropna()
        .assign(cpi_yoy_pct=lambda d: 100 * d["cpi_yoy"])
    )

    # 11) DF YoY desde 1996
    dt_ini_yoy = pd.Timestamp("1996-01-01")
    df_yoy_plot = (
        serie_trim[["REF_DATE", "cpi_trim_yoy_pct"]]
        .merge(serie_cpi_yoy[["REF_DATE", "cpi_yoy_pct"]], on="REF_DATE", how="left")
        .query("REF_DATE >= @dt_ini_yoy")
        .copy()
    )

    # 12) DF curto desde 2011
    dt_ini_curto = pd.Timestamp("2011-01-01")
    df_curto = serie_trim.query("REF_DATE >= @dt_ini_curto").copy()

    # 13) mês referência para componentes
    if args.mes_ref is None:
        mes_ref = pd.to_datetime(serie_trim["REF_DATE"].max())
    else:
        mes_ref = pd.to_datetime(args.mes_ref)

    df_mes = df_base.query("REF_DATE == @mes_ref").copy()
    det = detalhar_trim_mes(df_mes, col_prod)
    det_inc = det.loc[det["included"]].copy()

    # 14) figuras (4 painéis)
    fig_yoy = grafico_yoy_cpi_vs_trim(df_yoy_plot)
    fig_curto = grafico_curto_mom_saar_mm3(df_curto)
    fig_mom = grafico_componentes_mom(det_inc, top_n=30)
    fig_contrib = grafico_componentes_contrib(det_inc, top_n=30)

    subtitulo = (
        f"Pesos do basket: {data_basket_ref:%Y-%m} | "
        f"Mês dos componentes: {mes_ref:%Y-%m} | "
        f"SA: STL"
    )

    # 15) HTML (dashboard 2x2)
    out_html = saida_dir / "dashboard_cpi_trim_canada.html"
    salvar_dashboard_html(fig_yoy, fig_curto, fig_mom, fig_contrib, out_html, subtitulo=subtitulo)
    print(f"[OK] HTML salvo em: {out_html.resolve()}")


    out_pdf = saida_dir / "dashboard_cpi_trim_canada.pdf"
    exportar_dashboard_pdf(fig_yoy, fig_curto, fig_mom, fig_contrib, out_pdf)

    # 17) CSVs úteis (opcional)
    serie_trim.to_csv(saida_dir / "serie_cpi_trim.csv", index=False)
    det_inc.to_csv(saida_dir / f"componentes_{mes_ref:%Y-%m-01}.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
