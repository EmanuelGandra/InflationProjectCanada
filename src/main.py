"""
cpi_trim_canada.py

Projeto (estudo): replicar uma versão do CPI-Trim do Canadá (BoC) usando dados do Statistics Canada.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import stats_can
from stats_can import sc

from config import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR

from io_statcan import (
    carregar_tabelas_statcan,
    filtrar_canada,
    selecionar_componentes_55,
    escolher_pesos_55,
)
from seasonal import dessazonalizar_componentes
from trim import calcular_mom, cpi_trim_mes, detalhar_trim_mes
from metrics import calcular_metricas
from charts import (
    grafico_yoy_cpi_vs_trim,
    grafico_curto_mom_saar_mm3,
    grafico_componentes_mom,
    grafico_componentes_contrib,
)
from report import salvar_dashboard_html, exportar_dashboard_pdf


def configurar_warnings() -> None:
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


def main() -> int:
    configurar_warnings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--saida_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Pasta de saída (html/pdf)")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Pasta de cache dos zips do StatCan")
    parser.add_argument("--basket_ref", type=str, default=None, help="Data de referência dos pesos (YYYY-MM-01). Default: última disponível.")
    parser.add_argument("--mes_ref", type=str, default=None, help="Mês dos gráficos de componentes (YYYY-MM-01). Default: último mês do CPI-trim.")
    parser.add_argument("--gerar_pdf", action="store_true", help="(mantido por compat) se setado, tenta gerar PDF do dashboard")
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

    # 8) série CPI-trim (m/m)
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

    # 16) PDF (kaleido+reportlab) — mantém seu fluxo
    out_pdf = saida_dir / "dashboard_cpi_trim_canada.pdf"
    exportar_dashboard_pdf(fig_yoy, fig_curto, fig_mom, fig_contrib, out_pdf)

    # 17) CSVs úteis
    serie_trim.to_csv(saida_dir / "serie_cpi_trim.csv", index=False)
    det_inc.to_csv(saida_dir / f"componentes_{mes_ref:%Y-%m-01}.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
