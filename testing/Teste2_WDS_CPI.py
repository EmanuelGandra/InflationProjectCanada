from __future__ import annotations

import io
import zipfile
import requests
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# CONFIG
# =========================================================
WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

# 18-10-0004-01 (CPI levels)  -> full table zip: 18100004
# 18-10-0007-01 (CPI weights) -> full table zip: 18100007
PID_CPI_LEVELS = 18100004
PID_WEIGHTS    = 18100007

LANG = "en"
GEOGRAPHY_TARGET = "Canada"
TRIM_TAIL = 0.20

OUT_HTML = "cpi_trim_report.html"
OUT_PDF  = None  # ex.: "cpi_trim_report.pdf" (requer kaleido)
OUT_SERIES_CSV = "cpi_trim_series.csv"
OUT_WEIGHTS_CSV = "cpi_trim_weights.csv"
OUT_COMPONENTS_MAP = "components_map.csv"


# =========================================================
# DOWNLOAD HELPERS
# =========================================================
def wds_get_full_table_zip_url(product_id: int, lang: str = "en") -> str:
    url = f"{WDS_BASE}/getFullTableDownloadCSV/{product_id}/{lang}"
    headers = {"Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    obj = r.json()
    if obj.get("status") != "SUCCESS":
        raise RuntimeError(f"WDS failure: {obj}")
    return obj["object"]


def direct_full_table_zip_url(product_id: int, lang: str = "en") -> str:
    suffix = "eng" if lang == "en" else "fra"
    return f"https://www150.statcan.gc.ca/n1/tbl/csv/{product_id}-{suffix}.zip"


def read_statcan_full_table(product_id: int, lang: str = "en") -> pd.DataFrame:
    headers_dl = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

    try:
        zip_url = wds_get_full_table_zip_url(product_id, lang=lang)
        rz = requests.get(zip_url, headers=headers_dl, timeout=180)
        rz.raise_for_status()
    except Exception as e:
        print(f"[WARN] WDS falhou para PID={product_id} ({e}). Tentando download direto...")
        zip_url = direct_full_table_zip_url(product_id, lang=lang)
        rz = requests.get(zip_url, headers=headers_dl, timeout=180)
        rz.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(rz.content))
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError("ZIP sem CSV encontrado.")

    csv_name = max(csv_names, key=lambda n: z.getinfo(n).file_size)
    with z.open(csv_name) as f:
        # low_memory=False remove DtypeWarning por mixed types
        df = pd.read_csv(f, low_memory=False)

    return df

# =========================================================
# DF HELPERS
# =========================================================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    raise KeyError(f"Não achei coluna. candidatos={candidates}. colunas={list(df.columns)}")


def to_month_start(s: pd.Series) -> pd.DatetimeIndex:
    """
    Converte REF_DATE do StatCan para início do mês.
    CORRIGE o erro: to_timestamp("MS") não é suportado.
    """
    dt = pd.to_datetime(s, errors="coerce")
    p = pd.DatetimeIndex(dt).to_period("M")
    # 'how="start"' dá início do mês
    return p.to_timestamp(how="start")


# =========================================================
# Seasonal adjustment: STL em log(nível)
# =========================================================
def stl_sa_level(x: pd.Series, period: int = 12) -> pd.Series:
    x = x.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3 * period:
        return x
    lx = np.log(x)
    res = STL(lx, period=period, robust=True).fit()
    sa = np.exp(res.trend + res.resid)
    return sa


# =========================================================
# Trimmed mean (1 mês)
# =========================================================
def trimmed_mean_one_month(rates: pd.Series, weights: pd.Series, tail: float = 0.20) -> float:
    df = pd.DataFrame({"rate": rates, "w": weights}).dropna()
    if df.empty:
        return np.nan

    df = df.sort_values("rate", ascending=True).copy()
    df["cw"] = df["w"].cumsum()

    low_cut = tail
    high_cut = 1.0 - tail

    keep = (df["cw"] > low_cut) & (df["cw"] <= high_cut)
    dfk = df.loc[keep].copy()
    if dfk.empty:
        return np.nan

    return float(np.average(dfk["rate"], weights=dfk["w"]))

# =========================================================
# Pipeline: build trim series
# =========================================================
def build_trim_series(df_cpi: pd.DataFrame, df_w: pd.DataFrame, geography: str) -> tuple[pd.DataFrame, pd.Series]:
    # CPI levels
    col_date = pick_col(df_cpi, ["REF_DATE", "ref_date"])
    col_geo  = pick_col(df_cpi, ["GEO", "geo"])
    col_item = pick_col(df_cpi, ["Products and product groups", "Products and product groups (CPI)", "products and product groups"])
    col_val  = pick_col(df_cpi, ["VALUE", "value"])

    d = df_cpi.copy()
    d = d[d[col_geo].astype(str).str.strip().eq(geography)].copy()
    d["date"] = to_month_start(d[col_date])
    d["item"] = d[col_item].astype(str).str.strip()
    d["value"] = pd.to_numeric(d[col_val], errors="coerce")

    # Weights
    w_date = pick_col(df_w, ["REF_DATE", "ref_date"])
    w_geo  = pick_col(df_w, ["GEO", "geo"])
    w_item = pick_col(df_w, ["Products and product groups", "Products and product groups (CPI)", "products and product groups"])
    w_val  = pick_col(df_w, ["VALUE", "value"])

    wdf = df_w.copy()
    wdf = wdf[wdf[w_geo].astype(str).str.strip().eq(geography)].copy()
    wdf["date"] = to_month_start(wdf[w_date])
    wdf["item"] = wdf[w_item].astype(str).str.strip()
    wdf["w"] = pd.to_numeric(wdf[w_val], errors="coerce")

    last_w_date = wdf["date"].max()
    w_last = wdf[wdf["date"] == last_w_date].copy()
    w_last = w_last.groupby("item", as_index=True)["w"].mean()

    # normaliza pesos se vierem em %
    if w_last.sum() > 1.5:
        w_last = w_last / 100.0

    # Matriz de níveis (NSA)
    levels = (
        d.pivot_table(index="date", columns="item", values="value", aggfunc="mean")
        .sort_index()
    )

    # Match CPI x pesos
    common_items = sorted(set(levels.columns).intersection(set(w_last.index)))
    if len(common_items) != 55:
        print(f"[AVISO] Itens comuns CPI x pesos = {len(common_items)} (esperado=55).")
        pd.DataFrame({"item_in_cpi": sorted(levels.columns)}).to_csv(OUT_COMPONENTS_MAP, index=False)
        print(f"  -> Salvei {OUT_COMPONENTS_MAP} para você selecionar/confirmar os 55 subitens.")

    levels = levels[common_items].copy()
    weights = w_last.loc[common_items].copy()
    weights = weights / weights.sum()

    # Seasonal adjustment
    sa_levels = pd.DataFrame(index=levels.index)
    for c in levels.columns:
        sa = stl_sa_level(levels[c])
        sa_levels[c] = sa.reindex(levels.index)

    # MoM SA
    mom_sa = sa_levels.pct_change()

    # Trimmed mean (MoM SA)
    trim_vals = []
    for dt in mom_sa.index:
        trim_vals.append(trimmed_mean_one_month(mom_sa.loc[dt], weights, tail=TRIM_TAIL))
    trim_mom = pd.Series(trim_vals, index=mom_sa.index, name="CPI_trim_MoM_SA")

    # Derivadas
    out = pd.DataFrame(index=trim_mom.index)
    out["MoM_SA"] = trim_mom
    out["MoM_SAAR"] = (1.0 + out["MoM_SA"]) ** 12 - 1.0
    out["MM3_SAAR"] = ((1.0 + out["MoM_SA"]).rolling(3).apply(lambda x: np.prod(x), raw=True) ** 4) - 1.0
    out["YoY_SA"] = (1.0 + out["MoM_SA"]).rolling(12).apply(lambda x: np.prod(x), raw=True) - 1.0

    return out, weights

# =========================================================
# Report
# =========================================================
def make_report(out: pd.DataFrame, geography: str, html_path: str, pdf_path: str | None = None):
    df = out.dropna().copy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "CPI-trim MoM (SA)",
            "CPI-trim MoM SAAR (SA)",
            "CPI-trim MM3 SAAR (SA)",
            "CPI-trim YoY (SA)",
        ),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["MoM_SA"] * 100, name="MoM (SA)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MoM_SAAR"] * 100, name="MoM SAAR (SA)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df["MM3_SAAR"] * 100, name="MM3 SAAR (SA)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["YoY_SA"] * 100, name="YoY (SA)"), row=2, col=2)

    fig.update_layout(
        title=f"Canada CPI Trimmed Mean (CPI-trim) — {geography}",
        height=850,
        legend_orientation="h",
        legend_y=-0.15,
    )
    fig.update_yaxes(title_text="%")
    fig.update_xaxes(title_text="Date")

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>CPI-trim report</title>
      </head>
      <body>
        <h2>Bank of Canada core inflation — Trimmed mean (CPI-trim)</h2>
        <p><b>Geography:</b> {geography}</p>
        <p><b>Definition:</b> trim 20% (by weight) from bottom + 20% from top of monthly price changes; compute weighted mean of remaining 60%.</p>
        <p><b>Notes:</b> Seasonal adjustment via STL (approximation). Indirect tax adjustment not applied.</p>
        {fig.to_html(full_html=False, include_plotlyjs="cdn")}
      </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] HTML gerado: {html_path}")

    if pdf_path:
        fig.write_image(pdf_path)  # requer kaleido
        print(f"[OK] PDF gerado: {pdf_path}")

# =========================================================
# MAIN
# =========================================================
def main():
    print("Baixando tabelas do StatCan via WDS...")
    df_cpi = read_statcan_full_table(PID_CPI_LEVELS, lang=LANG)
    df_w   = read_statcan_full_table(PID_WEIGHTS, lang=LANG)

    print("Calculando CPI-trim (trimmed mean)...")
    out, weights = build_trim_series(df_cpi, df_w, geography=GEOGRAPHY_TARGET)

    out.to_csv(OUT_SERIES_CSV)
    weights.to_csv(OUT_WEIGHTS_CSV, header=["weight"])

    print(f"[OK] Série salva: {OUT_SERIES_CSV}")
    print(f"[OK] Pesos salvos: {OUT_WEIGHTS_CSV}")

    print("Gerando report...")
    make_report(out, geography=GEOGRAPHY_TARGET, html_path=OUT_HTML, pdf_path=OUT_PDF)

    print("Concluído.")


if __name__ == "__main__":
    main()
