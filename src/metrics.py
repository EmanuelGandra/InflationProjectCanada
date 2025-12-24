from __future__ import annotations

import numpy as np
import pandas as pd


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
