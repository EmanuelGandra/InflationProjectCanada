from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import STL

from config import COMPONENTES_55


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
