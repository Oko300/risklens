"""
server.py — RiskLens FastMCP server
Exposes compare_filings tool. Protects with ctxprotocol JWT middleware.
Returns stable, explicit output schema. Never silently suppresses failures.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
except ImportError:
    raise ImportError(
        "fastmcp is required. Install with: pip install fastmcp"
    )

from fetcher import fetch_two_filings
from extractor import extract_sections
from delta import compute_delta
from scorer import score_sections, DISCLAIMER

# ---------------------------------------------------------------------------
# MCP application
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="RiskLens",
    version="1.0.0",
    description=(
        "Compares the two most recent 10-Q or 10-K filings for a US public company, "
        "focusing on Risk Factors (Item 1A) and MD&A (Item 7). "
        "Outputs are labeled as estimates. Coverage gaps are disclosed."
    ),
)

TOOL_TIMEOUT = 55  # seconds — stay within 60s grant requirement


# ---------------------------------------------------------------------------
# Output schema (explicit, stable)
# ---------------------------------------------------------------------------

def _build_output(
    ticker: str,
    form_type: str,
    pipeline_success: bool,
    failure_reason: Optional[str],
    newer_meta: Optional[Any],
    older_meta: Optional[Any],
    newer_extraction: Optional[Any],
    older_extraction: Optional[Any],
    delta_result: Optional[Any],
    scoring_result: Optional[Any],
    elapsed_seconds: float,
) -> dict:
    """
    Build the canonical RiskLens output dict.
    Schema is versioned and must not be changed without updating schema_version.
    """

    def meta_dict(m) -> Optional[dict]:
        if m is None:
            return None
        return {
            "ticker": m.ticker,
            "cik": m.cik,
            "form_type": m.form_type,
            "accession_number": m.accession_number,
            "filing_date": m.filing_date,
            "report_date": m.report_date,
            "document_url": m.document_url,
            "fetch_success": m.fetch_success,
            "failure_reason": m.failure_reason,
            "html_byte_length": m.html_byte_length,
        }

    def extraction_dict(e) -> Optional[dict]:
        if e is None:
            return None

        def section_dict(s):
            return {
                "section_name": s.section_name,
                "item_label": s.item_label,
                "extraction_success": s.extraction_success,
                "method": s.method.value,
                "confidence_score": s.confidence_score,
                "char_count": s.char_count,
                "failure_reason": s.failure_reason,
                "coverage_gap_note": s.coverage_gap_note,
                # Text intentionally omitted from structured output — too large
                # Available as raw_text via separate tool if needed
            }

        return {
            "filing_accession": e.filing_accession,
            "filing_date": e.filing_date,
            "risk_factors": section_dict(e.risk_factors),
            "mda": section_dict(e.mda),
            "full_doc_char_count": e.full_doc_char_count,
            "known_gaps": e.known_gaps,
            "both_succeeded": e.both_succeeded,
            "any_succeeded": e.any_succeeded,
        }

    def delta_dict(d) -> Optional[dict]:
        if d is None:
            return None

        def section_delta_dict(sd):
            # Trim changes list for output — full list available separately
            top_changes = []
            for c in sd.changes[:50]:  # Top 50 changes
                if c.change_type in ("added", "removed", "rewritten"):
                    top_changes.append({
                        "type": c.change_type,
                        "older": (c.older_text or "")[:300],
                        "newer": (c.newer_text or "")[:300],
                        "similarity": c.similarity,
                    })

            return {
                "section_name": sd.section_name,
                "magnitude": sd.magnitude.value,
                "total_older_sentences": sd.total_older_sentences,
                "total_newer_sentences": sd.total_newer_sentences,
                "added_count": sd.added_count,
                "removed_count": sd.removed_count,
                "rewritten_count": sd.rewritten_count,
                "unchanged_count": sd.unchanged_count,
                "pct_changed": sd.pct_changed,
                "delta_success": sd.delta_success,
                "failure_reason": sd.failure_reason,
                "top_changes": top_changes,
            }

        return {
            "risk_factors": section_delta_dict(d.risk_factors),
            "mda": section_delta_dict(d.mda),
            "comparison_note": d.comparison_note,
        }

    def scoring_dict(s) -> Optional[dict]:
        if s is None:
            return None

        def section_score_dict(ss):
            def hit_list(hits):
                return [
                    {
                        "signal": h.signal,
                        "tier": h.tier,
                        "weight": h.weight,
                        "in_change": h.in_change,
                        "context": h.context[:200],
                    }
                    for h in hits
                ]

            return {
                "section_name": ss.section_name,
                "materiality": ss.materiality.value,
                "raw_score": ss.raw_score,
                "is_estimate": ss.is_estimate,
                "analyst_note": ss.analyst_note,
                "tier1_hits": hit_list(ss.tier1_hits),
                "tier2_hits": hit_list(ss.tier2_hits[:10]),
                "new_signals": hit_list(ss.new_signals),
                "removed_signals": hit_list(ss.removed_signals),
            }

        return {
            "risk_factors": section_score_dict(s.risk_factors),
            "mda": section_score_dict(s.mda),
            "overall_materiality": s.overall_materiality.value,
            "top_signals": s.top_signals,
            "scoring_success": s.scoring_success,
            "failure_reason": s.failure_reason,
            "disclaimer": s.disclaimer,
        }

    return {
        "schema_version": "1.0",
        "tool": "compare_filings",
        "ticker": ticker,
        "form_type": form_type,
        "pipeline_success": pipeline_success,
        "failure_reason": failure_reason,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "newer_filing": meta_dict(newer_meta),
        "older_filing": meta_dict(older_meta),
        "newer_extraction": extraction_dict(newer_extraction),
        "older_extraction": extraction_dict(older_extraction),
        "delta": delta_dict(delta_result),
        "scoring": scoring_dict(scoring_result),
        "disclaimer": DISCLAIMER,
        "coverage_gap_disclosure": (
            "RiskLens only analyzes 10-Q and 10-K filings. "
            "Only Risk Factors (Item 1A) and MD&A (Item 7) sections are compared. "
            "Only the two most recent filings are compared. "
            "XBRL inline filings, exhibits, and amendments are not separately processed. "
            "Foreign private issuers (20-F) are not supported. "
            "Extraction may fail on heavily structured or image-based filings."
        ),
    }


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@mcp.tool()
async def compare_filings(
    ticker: str,
    form_type: Literal["10-Q", "10-K"] = "10-Q",
) -> dict:
    """
    Compare the two most recent SEC filings (10-Q or 10-K) for a US public company.
    Analyzes Risk Factors (Item 1A) and MD&A (Item 7) only.
    Returns structured delta analysis and materiality scores (labeled as estimates).

    Args:
        ticker: US stock ticker symbol (e.g. AAPL, MSFT, TSLA)
        form_type: Filing type — "10-Q" (quarterly) or "10-K" (annual). Default: 10-Q
    """
    start = time.monotonic()

    # Input validation
    ticker = ticker.upper().strip()
    if not ticker or not ticker.replace("-", "").isalpha():
        raise ToolError(f"Invalid ticker: {ticker!r}. Must be alphabetic (e.g. AAPL).")
    if form_type not in ("10-Q", "10-K"):
        raise ToolError("form_type must be '10-Q' or '10-K'.")

    # Run the pipeline with overall timeout guard
    try:
        result = await asyncio.wait_for(
            _run_pipeline(ticker, form_type),
            timeout=TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return _build_output(
            ticker=ticker,
            form_type=form_type,
            pipeline_success=False,
            failure_reason=f"Pipeline timed out after {TOOL_TIMEOUT}s",
            newer_meta=None, older_meta=None,
            newer_extraction=None, older_extraction=None,
            delta_result=None, scoring_result=None,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return _build_output(
            ticker=ticker,
            form_type=form_type,
            pipeline_success=False,
            failure_reason=f"Unexpected pipeline error: {exc}",
            newer_meta=None, older_meta=None,
            newer_extraction=None, older_extraction=None,
            delta_result=None, scoring_result=None,
            elapsed_seconds=elapsed,
        )

    elapsed = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def _run_pipeline(ticker: str, form_type: str) -> dict:
    start = time.monotonic()

    # Step 1: Fetch
    fetch_result = await fetch_two_filings(ticker, form_type)  # type: ignore

    if not fetch_result.pipeline_success:
        elapsed = time.monotonic() - start
        return _build_output(
            ticker=ticker, form_type=form_type,
            pipeline_success=False,
            failure_reason=fetch_result.failure_reason,
            newer_meta=fetch_result.newer, older_meta=fetch_result.older,
            newer_extraction=None, older_extraction=None,
            delta_result=None, scoring_result=None,
            elapsed_seconds=elapsed,
        )

    # Step 2: Extract sections from both filings
    newer_extraction = extract_sections(
        fetch_result.newer_html or "",
        accession=fetch_result.newer.accession_number if fetch_result.newer else "",
        filing_date=fetch_result.newer.filing_date if fetch_result.newer else "",
    )
    older_extraction = extract_sections(
        fetch_result.older_html or "",
        accession=fetch_result.older.accession_number if fetch_result.older else "",
        filing_date=fetch_result.older.filing_date if fetch_result.older else "",
    )

    # Step 3: Delta
    delta_result = compute_delta(
        older_risk=older_extraction.risk_factors.text,
        newer_risk=newer_extraction.risk_factors.text,
        older_mda=older_extraction.mda.text,
        newer_mda=newer_extraction.mda.text,
    )

    # Step 4: Score
    scoring_result = score_sections(
        newer_risk_text=newer_extraction.risk_factors.text,
        older_risk_text=older_extraction.risk_factors.text,
        newer_mda_text=newer_extraction.mda.text,
        older_mda_text=older_extraction.mda.text,
        risk_delta=delta_result.risk_factors,
        mda_delta=delta_result.mda,
    )

    elapsed = time.monotonic() - start

    # Pipeline is "successful" even if extraction partially fell back —
    # partial data is better than no data. Failures are flagged in sub-objects.
    overall_success = fetch_result.pipeline_success

    return _build_output(
        ticker=ticker, form_type=form_type,
        pipeline_success=overall_success,
        failure_reason=None,
        newer_meta=fetch_result.newer,
        older_meta=fetch_result.older,
        newer_extraction=newer_extraction,
        older_extraction=older_extraction,
        delta_result=delta_result,
        scoring_result=scoring_result,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# ctxprotocol JWT middleware
# ---------------------------------------------------------------------------

def _build_auth_middleware():
    """
    Attach ctxprotocol JWT verification if CTX_PUBLIC_KEY is set in env.
    Falls back to no-auth (dev mode) with a warning if key is absent.
    """
    ctx_key = os.getenv("CTX_PUBLIC_KEY")
    if not ctx_key:
        print(
            "[RiskLens] WARNING: CTX_PUBLIC_KEY not set. "
            "Running without ctxprotocol auth (dev mode only)."
        )
        return None

    try:
        from ctxprotocol import JWTMiddleware
        return JWTMiddleware(public_key=ctx_key)
    except ImportError:
        print(
            "[RiskLens] WARNING: ctxprotocol not installed. "
            "Running without auth middleware."
        )
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    middleware = _build_auth_middleware()
    if middleware:
        mcp.add_middleware(middleware)

    print(f"[RiskLens] Starting MCP server (tool timeout: {TOOL_TIMEOUT}s)")
    mcp.run()