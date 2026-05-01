"""
app.py — RiskLens public REST API
Exposes a single GET /fetch endpoint for browser and demo access.
Runs alongside the existing FastMCP server — does NOT modify it.

Deploy on Railway. Binds to 0.0.0.0:PORT.
"""

import os
import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from fetcher import fetch_two_filings

app = FastAPI(
    title="RiskLens Public API",
    description="Public REST interface for RiskLens EDGAR filing comparison.",
    version="1.0.0",
)


@app.get("/")
async def health():
    """Health check — confirms the API is running."""
    return {"status": "ok", "service": "RiskLens Public API", "version": "1.0.0"}


@app.get("/fetch")
async def fetch(
    ticker: str = Query(..., description="US stock ticker, e.g. AAPL"),
    form_type: str = Query("10-Q", description="10-Q or 10-K"),
):
    """
    Fetch and compare the two most recent SEC filings for a ticker.
    Returns structured metadata and pipeline status.
    """
    if form_type not in ("10-Q", "10-K"):
        return JSONResponse(
            status_code=400,
            content={"error": "form_type must be '10-Q' or '10-K'"},
        )

    result = await fetch_two_filings(ticker.upper().strip(), form_type)

    def meta(m):
        if m is None:
            return None
        return {
            "ticker":           m.ticker,
            "cik":              m.cik,
            "form_type":        m.form_type,
            "accession_number": m.accession_number,
            "filing_date":      m.filing_date,
            "report_date":      m.report_date,
            "document_url":     m.document_url,
            "fetch_success":    m.fetch_success,
            "failure_reason":   m.failure_reason,
            "html_byte_length": m.html_byte_length,
        }

    return {
        "ticker":           result.ticker,
        "cik":              result.cik,
        "form_type":        result.form_type,
        "pipeline_success": result.pipeline_success,
        "failure_reason":   result.failure_reason,
        "newer_filing":     meta(result.newer),
        "older_filing":     meta(result.older),
        "coverage_gap_note": result.coverage_gap_note,
        "note": (
            "This endpoint returns filing metadata only. "
            "Full section extraction and delta analysis are available via the MCP tool."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)