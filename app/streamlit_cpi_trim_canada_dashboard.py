"""
streamlit_cpi_trim_canada_dashboard.py

Dashboard (Streamlit) para replicar uma versão do CPI-Trim do Canadá (BoC)
com dados do Statistics Canada (via pacote stats_can).

O que você consegue fazer no app
- Escolher intervalo de datas (início e fim) para os gráficos
- Escolher o mês de referência dos componentes (barras)
- Ajustar o % de corte nas caudas (trim) — ex.: 20%/20% (padrão)
- Selecionar a data de pesos (basket_ref), caso queira “fixar” o basket

Estrutura
- Tab 1: Séries (YoY / Curto prazo / Short-term momentum)
- Tab 2: Componentes (MoM SA e contribuição em bps) para um mês escolhido

Requisitos
pip install streamlit pandas numpy statsmodels plotly stats_can

Como rodar
streamlit run streamlit_cpi_trim_canada_dashboard.py

Obs
- A dessazonalização aqui é STL por componente (robust=True). Não trata impostos.
- O stats_can pode emitir warnings internos (FutureWarning). Eu filtro no app.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import STL

import stats_can
from stats_can import sc


# =========================================================
# CONFIG: warnings (console limpo)
# =========================================================
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
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# CONFIG VISUAL (paleta e layout)
# =========================================================
COR_PRINCIPAL = "#c00000"   # destaque
COR_SECUNDARIA = "#5b89c1"  # secundária
GRID_Y = "rgba(0,0,0,0.10)"

# =========================================================
# PATHS (app/ -> raiz do projeto)
# =========================================================
APP_DIR = Path(__file__).resolve().parent          # .../app
PROJECT_ROOT = APP_DIR.parents[0]                  # .../ (diretório atual onde está app/)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR  = PROJECT_ROOT / "output"

DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# (opcional) cache do Streamlit fora do app/
STREAMLIT_CACHE_DIR = DEFAULT_OUT_DIR / "_st_cache"
STREAMLIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import os
os.environ["STREAMLIT_CACHE_DIR"] = str(STREAMLIT_CACHE_DIR)

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



def add_custom_css():
    st.markdown(
        """
        <style>
        /* ===== BaseWeb Select / Streamlit selectbox-multiselect ===== */

        /* Valor selecionado (o "2024-01" que aparece no campo) */
        div[data-baseweb="select"] [role="combobox"],
        div[data-baseweb="select"] [role="combobox"] * {
            color: #000 !important;
            -webkit-text-fill-color: #000 !important; /* ajuda no Chrome */
        }

        /* Input do combobox (quando existe <input role="combobox">) */
        div[data-baseweb="select"] input[role="combobox"]{
            color: #000 !important;
            -webkit-text-fill-color: #000 !important;
        }

        /* Itens do dropdown (lista aberta) */
        div[role="listbox"] [role="option"],
        div[role="listbox"] [role="option"] * {
            color: #000 !important;
            -webkit-text-fill-color: #000 !important;
        }

        /* Alguns temas colocam placeholder em opacity; força preto também */
        div[data-baseweb="select"] [aria-label]{
            color: #000 !important;
            -webkit-text-fill-color: #000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()


# =========================================================
# HELPERS: dados StatCan
# =========================================================
@st.cache_data(show_spinner=False)
def carregar_tabelas_statcan(cache_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa (ou usa cache) das tabelas do StatCan via stats_can."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    pid_cpi = "18-10-0004-13"
    pid_pesos = "18-10-0007-01"

    df_cpi = sc.zip_table_to_dataframe(pid_cpi, path=cache_path)
    df_pesos = sc.zip_table_to_dataframe(pid_pesos, path=cache_path)

    # garantir datetime (sem errors='ignore')
    df_cpi["REF_DATE"] = pd.to_datetime(df_cpi["REF_DATE"], errors="coerce")
    df_pesos["REF_DATE"] = pd.to_datetime(df_pesos["REF_DATE"], errors="coerce")

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
    """Seleciona pesos (55 linhas) e normaliza para somar 1."""
    df_basket = df_pesos_can.query("REF_DATE == @data_ref").copy()

    df_55 = (
        df_basket
        .query("`Price period of weight` == @price_period")
        .query("`Geographic distribution of weight` == @geo_dist")
        .query(f"`{col_prod}` in @COMPONENTES_55")
        .copy()
    )

    df_55["peso"] = df_55["VALUE"].astype(float)
    soma = float(df_55["peso"].sum())
    if soma <= 0 or np.isnan(soma):
        raise RuntimeError("Soma dos pesos veio zero/NaN. Checar filtros de pesos.")
    df_55["peso"] = df_55["peso"] / soma
    return df_55


# =========================================================
# AJUSTE SAZONAL (STL)
# =========================================================
def ajuste_sazonal_stl(serie: pd.Series, periodo: int = 12) -> pd.Series:
    serie = serie.dropna()
    if len(serie) < 2 * periodo:
        return serie
    stl = STL(serie, period=periodo, robust=True)
    res = stl.fit()
    return serie - res.seasonal


@st.cache_data(show_spinner=False)
def dessazonalizar_componentes(df_cpi_55: pd.DataFrame, col_prod: str) -> pd.DataFrame:
    """Dessazonaliza item a item e devolve: REF_DATE | produto | indice_sa"""
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


def calcular_mom(df_sa: pd.DataFrame, col_prod: str) -> pd.DataFrame:
    df = df_sa.sort_values([col_prod, "REF_DATE"]).copy()
    df["mom"] = df.groupby(col_prod)["indice_sa"].pct_change()
    return df


# =========================================================
# CPI-TRIM (núcleo)
# =========================================================
def cpi_trim_mes(df_mes: pd.DataFrame, lim_inf: float, lim_sup: float) -> float:
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

    denom = float(df["peso_efetivo"].sum())
    if denom <= 0 or np.isnan(denom):
        return np.nan

    trim_mom = float((df["mom"] * df["peso_efetivo"]).sum() / denom)
    return trim_mom


def detalhar_trim_mes(df_mes: pd.DataFrame, col_prod: str, lim_inf: float, lim_sup: float) -> pd.DataFrame:
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
    soma_eff = float(df.loc[df["included"], "peso_efetivo"].sum())

    df["share"] = np.where(df["included"] & (soma_eff > 0), df["peso_efetivo"] / soma_eff, 0.0)
    df["mom_pct"] = df["mom"] * 100
    df["contrib_bps"] = (df["mom"] * df["share"]) * 10000

    return df.rename(columns={col_prod: "componente"})


def calcular_metricas(serie: pd.DataFrame) -> pd.DataFrame:
    s = serie.sort_values("REF_DATE").copy()

    # 1m SAAR
    s["cpi_trim_mom_saar"] = (1 + s["cpi_trim_mom"])**12 - 1

    # MM3 (média no MoM) e anualização por composição (capitalizando)
    s["cpi_trim_mm3_mom"] = s["cpi_trim_mom"].rolling(3).mean()
    s["cpi_trim_mm3_saar"] = (1 + s["cpi_trim_mm3_mom"])**12 - 1

    # YoY composto
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
        margin=dict(l=70, r=30, t=70, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
        height=430,
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
        tickangle=0,
        showgrid=False,
        zeroline=False,
    )
    return fig


def grafico_yoy_cpi_vs_trim(df_yoy: pd.DataFrame) -> go.Figure:
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

    return _layout_base(fig, "Canada Inflation — CPI vs CPI-Trim (YoY)", "Percent")


def grafico_curto_mom_saar_mm3(df_curto: pd.DataFrame) -> go.Figure:
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

    return _layout_base(fig, "CPI-Trim — MoM SAAR vs MM3", "Percent (annualized)")


def grafico_short_term_momentum(df: pd.DataFrame) -> go.Figure:
    """Short-term momentum: YoY vs MM3 (annualized)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["REF_DATE"], y=df["cpi_trim_yoy_pct"],
        mode="lines", name="YoY",
        line=dict(color=COR_PRINCIPAL, width=2.7),
    ))
    fig.add_trace(go.Scatter(
        x=df["REF_DATE"], y=df["cpi_trim_mm3_saar_pct"],
        mode="lines", name="MM3 (annualized)",
        line=dict(color=COR_SECUNDARIA, width=2.2, dash="dash"),
    ))

    for y in [1, 2, 3]:
        fig.add_hline(y=y, line_width=1, line_dash="dot", opacity=0.25)

    return _layout_base(fig, "CPI-Trim — YoY vs Short-term Momentum", "Percent")


def grafico_componentes_mom(det_inc: pd.DataFrame, top_n: int = 30) -> go.Figure:
    mom = det_inc[["componente", "mom_pct"]].sort_values("mom_pct", ascending=False).copy()

    # metade top positivos + metade top negativos
    mom_pos = mom[mom["mom_pct"] >= 0].head(max(1, top_n // 2))
    mom_neg = mom[mom["mom_pct"] < 0].tail(max(1, top_n // 2))
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
        margin=dict(l=240, r=30, t=70, b=60),
        height=520,
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
        margin=dict(l=240, r=30, t=70, b=60),
        height=520,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_Y, zeroline=False, title="bps")
    fig.update_yaxes(showgrid=False, automargin=True, title="")
    return fig


# =========================================================
# PIPELINE principal (base) — cacheada
# =========================================================
@st.cache_data(show_spinner=True)
def preparar_base(cache_dir: str) -> dict:
    """\
    Prepara e devolve um pacote com:
    - df_cpi_can (Canada)
    - df_pesos_can (Canada)
    - df_cpi_55
    - df_sa (dessazonalizado por item)
    - df_sa_mom
    - serie_cpi_yoy (headline)
    - meses_disponiveis_base (para componentes)
    - datas_pesos_disponiveis (basket_ref)
    """
    df_cpi_bruto, df_pesos_bruto = carregar_tabelas_statcan(cache_dir)

    df_cpi_can = filtrar_canada(df_cpi_bruto)
    df_pesos_can = filtrar_canada(df_pesos_bruto)

    col_prod = "Products and product groups"
    df_cpi_55 = selecionar_componentes_55(df_cpi_can, col_prod)

    df_sa = dessazonalizar_componentes(df_cpi_55, col_prod)
    df_sa_mom = calcular_mom(df_sa, col_prod)

    # headline YoY (All-items)
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

    meses_disponiveis_base = (
        df_sa_mom["REF_DATE"].dropna().drop_duplicates().sort_values().tolist()
    )

    datas_pesos_disponiveis = (
        df_pesos_can["REF_DATE"].dropna().drop_duplicates().sort_values().tolist()
    )

    return {
        "col_prod": col_prod,
        "df_cpi_can": df_cpi_can,
        "df_pesos_can": df_pesos_can,
        "df_cpi_55": df_cpi_55,
        "df_sa_mom": df_sa_mom,
        "serie_cpi_yoy": serie_cpi_yoy,
        "meses": meses_disponiveis_base,
        "datas_pesos": datas_pesos_disponiveis,
    }


def montar_serie_trim(
    df_sa_mom: pd.DataFrame,
    df_pesos_55: pd.DataFrame,
    col_prod: str,
    corte_cauda_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """\
    - Merge MoM + peso
    - Calcula série CPI-Trim (m/m) com limiares derivados do corte
    Retorna:
      (serie_trim_com_metricas, df_base_merge)
    """
    lim_inf = corte_cauda_pct
    lim_sup = 1.0 - corte_cauda_pct

    df_base = df_sa_mom.merge(
        df_pesos_55[[col_prod, "peso"]],
        on=col_prod,
        how="left",
    )

    if df_base["peso"].isna().any():
        raise RuntimeError("Peso faltando após merge (55 componentes).")

    df_small = df_base[["REF_DATE", "mom", "peso"]].copy()

    # calcula o trim mês a mês
    serie_trim_gb = df_small.groupby("REF_DATE", sort=True, group_keys=False)

    # pandas mais novos: pode usar include_groups=False para evitar FutureWarning
    try:
        serie_trim = (
            serie_trim_gb
            .apply(lambda g: cpi_trim_mes(g, lim_inf=lim_inf, lim_sup=lim_sup), include_groups=False)
            .rename("cpi_trim_mom")
            .reset_index()
            .sort_values("REF_DATE")
        )
    except TypeError:
        serie_trim = (
            serie_trim_gb
            .apply(lambda g: cpi_trim_mes(g, lim_inf=lim_inf, lim_sup=lim_sup))
            .rename("cpi_trim_mom")
            .reset_index()
            .sort_values("REF_DATE")
        )

    serie_trim = calcular_metricas(serie_trim)
    return serie_trim, df_base


# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(page_title="CPI-Trim Canadá — Dashboard", layout="wide")

st.title("CPI-Trim Canadá — Dashboard")
st.caption("Dados: Statistics Canada (stats_can) | SA: STL por componente")

with st.sidebar:
    st.subheader("Filtros")

    cache_dir = str(DEFAULT_DATA_DIR)

    pacote = preparar_base(cache_dir)
    col_prod = pacote["col_prod"]

    # basket_ref: default último disponível
    datas_pesos = pacote["datas_pesos"]
    data_basket_default = datas_pesos[-1]

    # seletores
    data_basket_ref = st.selectbox(
        "Data dos pesos (basket_ref)",
        options=datas_pesos,
        index=len(datas_pesos) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
    )

    corte_pct = st.slider(
        "% corte em cada cauda (trim)",
        min_value=5,
        max_value=30,
        value=20,
        step=1,
        help="Ex.: 20 significa cortar 20% em cada cauda (fica o miolo 60%).",
    )
    corte_cauda = float(corte_pct) / 100.0

    # período dos gráficos
    meses = pacote["meses"]
    data_min = pd.Timestamp(meses[0]).date()
    data_max = pd.Timestamp(meses[-1]).date()

    inicio, fim = st.date_input(
        "Intervalo de datas dos gráficos",
        value=(max(data_min, pd.Timestamp("1996-01-01").date()), data_max),
        min_value=data_min,
        max_value=data_max,
    )
    dt_ini = pd.Timestamp(inicio)
    dt_fim = pd.Timestamp(fim)
    if dt_ini > dt_fim:
        st.error("Data inicial maior que a final. Ajuste o intervalo.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Componentes")

    # mês para componentes
    mes_componentes = st.selectbox(
        "Mês dos componentes",
        options=meses,
        index=len(meses) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
    )

    top_n = st.slider(
        "Quantidade de componentes nas barras (top N)",
        min_value=10,
        max_value=55,
        value=30,
        step=5,
    )


# 1) Pesos 55
try:
    df_pesos_55 = escolher_pesos_55(
        pacote["df_pesos_can"],
        col_prod=col_prod,
        data_ref=pd.to_datetime(data_basket_ref),
    )
except Exception as e:
    st.error(f"Erro ao selecionar pesos: {e}")
    st.stop()

# 2) Série trim (recalcula quando mudar trim ou basket)
try:
    serie_trim, df_base = montar_serie_trim(
        pacote["df_sa_mom"],
        df_pesos_55,
        col_prod=col_prod,
        corte_cauda_pct=corte_cauda,
    )
except Exception as e:
    st.error(f"Erro ao calcular CPI-Trim: {e}")
    st.stop()

# 3) DF YoY (CPI vs Trim)
dt_ini_yoy_min = pd.Timestamp("1996-01-01")
serie_cpi_yoy = pacote["serie_cpi_yoy"]

# usa intervalo do usuário, mas nunca abaixo de 1996
filtro_ini_yoy = max(dt_ini, dt_ini_yoy_min)

df_yoy_plot = (
    serie_trim[["REF_DATE", "cpi_trim_yoy_pct"]]
    .merge(serie_cpi_yoy[["REF_DATE", "cpi_yoy_pct"]], on="REF_DATE", how="left")
    .query("REF_DATE >= @filtro_ini_yoy and REF_DATE <= @dt_fim")
    .copy()
)

# 4) Curto prazo (MoM SAAR vs MM3)
df_curto = (
    serie_trim
    .query("REF_DATE >= @dt_ini and REF_DATE <= @dt_fim")
    .dropna(subset=["cpi_trim_mom_saar_pct", "cpi_trim_mm3_saar_pct"])
    .copy()
)

# 5) Momentum (YoY vs MM3 annualized)
df_momentum = (
    serie_trim
    .query("REF_DATE >= @dt_ini and REF_DATE <= @dt_fim")
    .dropna(subset=["cpi_trim_yoy_pct", "cpi_trim_mm3_saar_pct"])
    .copy()
)

# 6) Componentes do mês selecionado
mes_ref = pd.to_datetime(mes_componentes)
df_mes = df_base.query("REF_DATE == @mes_ref").copy()

lim_inf = corte_cauda
lim_sup = 1.0 - corte_cauda

det = detalhar_trim_mes(df_mes, col_prod=col_prod, lim_inf=lim_inf, lim_sup=lim_sup)
det_inc = det.loc[det.get("included", False)].copy() if not det.empty else det


# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["Séries", "Componentes"])

with tab1:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("YoY — CPI vs CPI-Trim")
        if df_yoy_plot.empty:
            st.info("Sem dados no intervalo selecionado.")
        else:
            st.plotly_chart(grafico_yoy_cpi_vs_trim(df_yoy_plot), use_container_width=True)

    with colB:
        st.subheader("Curto prazo — MoM SAAR vs MM3")
        if df_curto.empty:
            st.info("Sem dados no intervalo selecionado.")
        else:
            st.plotly_chart(grafico_curto_mom_saar_mm3(df_curto), use_container_width=True)

    st.subheader("Short-term momentum")
    if df_momentum.empty:
        st.info("Sem dados suficientes para o gráfico de momentum nesse intervalo.")
    else:
        st.plotly_chart(grafico_short_term_momentum(df_momentum), use_container_width=True)

    with st.expander("Ver série (amostra)"):
        st.dataframe(
            serie_trim[["REF_DATE", "cpi_trim_mom_pct", "cpi_trim_mom_saar_pct", "cpi_trim_mm3_saar_pct", "cpi_trim_yoy_pct"]]
            .query("REF_DATE >= @dt_ini and REF_DATE <= @dt_fim")
            .tail(24)
            .reset_index(drop=True)
        )


with tab2:
    st.subheader(f"Componentes do mês — {mes_ref:%Y-%m}")
    st.caption(f"Trim: {corte_pct}% / {corte_pct}% | Basket_ref: {pd.to_datetime(data_basket_ref):%Y-%m}")

    if det_inc.empty:
        st.info("Não consegui montar os componentes desse mês (sem dados ou denom=0).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(grafico_componentes_mom(det_inc, top_n=top_n), use_container_width=True)
        with col2:
            st.plotly_chart(grafico_componentes_contrib(det_inc, top_n=top_n), use_container_width=True)

        with st.expander("Tabela completa (componentes incluídos)"):
            tabela = (
                det_inc[["componente", "mom_pct", "contrib_bps", "share", "peso", "peso_efetivo"]]
                .sort_values("contrib_bps", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(tabela)


# =========================================================
# Rodapé
# =========================================================
st.markdown("---")
st.caption(
    "Dica: se ficar lento, é normal na primeira execução (dessazonalização STL por 55 componentes). "
    "Depois o Streamlit cacheia e fica bem mais rápido."
)

add_custom_css()

