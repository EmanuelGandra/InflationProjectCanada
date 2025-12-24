from __future__ import annotations

import io
from pathlib import Path

import plotly.graph_objects as go

from config import COR_PRINCIPAL, COR_SECUNDARIA


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
        import kaleido  # noqa: F401
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.utils import ImageReader
    except Exception:
        print("[Aviso] PDF do dashboard não gerado. Instale: pip install -U kaleido reportlab")
        return

    scale = 2.0
    png1 = fig_yoy.to_image(format="png", scale=scale)
    png2 = fig_curto.to_image(format="png", scale=scale)
    png3 = fig_mom.to_image(format="png", scale=scale)
    png4 = fig_contrib.to_image(format="png", scale=scale)

    img1 = ImageReader(io.BytesIO(png1))
    img2 = ImageReader(io.BytesIO(png2))
    img3 = ImageReader(io.BytesIO(png3))
    img4 = ImageReader(io.BytesIO(png4))

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

            page.goto(out_html.resolve().as_uri(), wait_until="networkidle")

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
