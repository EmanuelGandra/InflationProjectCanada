from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import stats_can
from stats_can import sc

from config import COMPONENTES_55


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
