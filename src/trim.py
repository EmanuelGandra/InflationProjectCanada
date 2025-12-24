from __future__ import annotations

import numpy as np
import pandas as pd


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
