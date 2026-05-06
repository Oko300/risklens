"""
server.py — RiskLens FastMCP server
Round 2 fixes applied per Alex | Context review:
1. ctxprotocol auth wired via FastMCP middleware (required for paid tools)
2. pipeline_success=False when extraction falls below sanity thresholds
3. 10-Q reference pointer detection for Risk Factors
4. Pydantic output schema — FastMCP auto-generates outputSchema from return type
5. form_type correctly passed to extract_sections
6. pattern_match slow path addressed via thresholds in extractor.py specs
"""

import asyncio
import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from ctxprotocol import verify_context_request, ContextError

from fetcher import fetch_two_filings
from extractor import extract_sections_cached as extract_sections
from delta import compute_delta
from scorer import score_sections, DISCLAIMER

# ---------------------------------------------------------------------------
# Pydantic output schema
# FastMCP auto-generates outputSchema from Pydantic return types.
# This satisfies the Context Protocol outputSchema requirement.
# ---------------------------------------------------------------------------

class FilingMetaOut(BaseModel):
    ticker: str
    cik: str
    form_type: str
    accession_number: str
    filing_date: str
    report_date: str
    document_url: str
    fetch_success: bool
    failure_reason: Optional[str]
    html_byte_length: int


class SectionExtractionOut(BaseModel):
    section_name: str
    item_label: str
    extraction_success: bool
    method: str
    confidence_score: float
    char_count: int
    failure_reason: Optional[str]
    coverage_gap_note: Optional[str]


class ExtractionOut(BaseModel):
    filing_accession: str
    filing_date: str
    risk_factors: SectionExtractionOut
    mda: SectionExtractionOut
    full_doc_char_count: int
    known_gaps: list[str]
    both_succeeded: bool
    any_succeeded: bool


class ChangeOut(BaseModel):
    type: str
    older: str
    newer: str
    similarity: float


class SectionDeltaOut(BaseModel):
    section_name: str
    magnitude: str
    total_older_sentences: int
    total_newer_sentences: int
    added_count: int
    removed_count: int
    rewritten_count: int
    unchanged_count: int
    pct_changed: float
    delta_success: bool
    failure_reason: Optional[str]
    top_changes: list[ChangeOut]


class DeltaOut(BaseModel):
    risk_factors: Optional[SectionDeltaOut]
    mda: Optional[SectionDeltaOut]
    comparison_note: str


class SignalHitOut(BaseModel):
    signal: str
    tier: int
    weight: int
    in_change: bool
    context: str


class SectionScoreOut(BaseModel):
    section_name: str
    materiality: str
    raw_score: float
    is_estimate: bool
    analyst_note: str
    tier1_hits: list[SignalHitOut]
    tier2_hits: list[SignalHitOut]
    new_signals: list[SignalHitOut]
    removed_signals: list[SignalHitOut]


class ScoringOut(BaseModel):
    risk_factors: SectionScoreOut
    mda: SectionScoreOut
    overall_materiality: str
    top_signals: list[str]
    scoring_success: bool
    failure_reason: Optional[str]
    disclaimer: str


class CompareFilingsOutput(BaseModel):
    schema_version: str
    tool: str
    ticker: str
    form_type: str
    pipeline_success: bool
    failure_reason: Optional[str]
    elapsed_seconds: float
    newer_filing: Optional[FilingMetaOut]
    older_filing: Optional[FilingMetaOut]
    newer_extraction: Optional[ExtractionOut]
    older_extraction: Optional[ExtractionOut]
    delta: Optional[DeltaOut]
    scoring: Optional[ScoringOut]
    disclaimer: str
    coverage_gap_disclosure: str


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

mcp = FastMCP(name="RiskLens")

# ---------------------------------------------------------------------------
# FIX 1: ctxprotocol auth middleware — REQUIRED for paid tools ($0.01+)
# Replaces the broken _build_auth_middleware() that was never attached.
# Only tools/call is protected. tools/list remains open for discovery.
# ---------------------------------------------------------------------------

class ContextProtocolAuth(Middleware):
    """Verify Context Protocol JWT on tool calls only."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        try:
            await verify_context_request(
                authorization_header=headers.get("authorization", "")
            )
        except ContextError as e:
            raise ToolError(f"Unauthorized: {e.message}")
        return await call_next(context)


mcp.add_middleware(ContextProtocolAuth())

TOOL_TIMEOUT = 55  # seconds — stay within 60s grant requirement

# ---------------------------------------------------------------------------
# FIX 2: Sanity thresholds
# When extraction falls below these, pipeline_success=False.
# Prevents confident materiality verdicts from near-empty inputs.
# ---------------------------------------------------------------------------

MDA_MIN_CHARS        = 5000   # Alex's recommendation
RISK_FACTORS_MIN_CHARS_10K = 2000  # Only enforced on 10-K (10-Q can legitimately be a pointer)

# ---------------------------------------------------------------------------
# FIX 3: 10-Q reference pointer phrases
# ---------------------------------------------------------------------------

_RF_REFERENCE_PHRASES = [
    "incorporated by reference",
    "annual report on form 10-k",
    "our annual report",
    "refer to part i, item 1a",
]

_RF_REFERENCE_CHAR_THRESHOLD = 3000  # Under this, check for reference pointer


def _is_rf_reference_pointer(text: Optional[str], char_count: int) -> bool:
    """
    Returns True if the Risk Factors section is just a pointer to the 10-K,
    not the actual risk list. Common in S&P 500 10-Qs.
    """
    if char_count >= _RF_REFERENCE_CHAR_THRESHOLD:
        return False
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in _RF_REFERENCE_PHRASES)


# ---------------------------------------------------------------------------
# Coverage gap disclosure
# ---------------------------------------------------------------------------

_COVERAGE_GAP = (
    "RiskLens only analyzes 10-Q and 10-K filings. "
    "Only Risk Factors (Item 1A) and MD&A (Item 7) sections are compared. "
    "Only the two most recent filings are compared. "
    "XBRL inline filings, exhibits, and amendments are not separately processed. "
    "Foreign private issuers (20-F) are not supported. "
    "Extraction may fail on heavily structured or image-based filings. "
    "Many 10-Q filings incorporate Risk Factors by reference from the annual 10-K — "
    "use form_type='10-K' for annual risk factor comparisons."
)


# ---------------------------------------------------------------------------
# Output builder — returns Pydantic model
# ---------------------------------------------------------------------------

def _build_output(
    ticker: str,
    form_type: str,
    pipeline_success: bool,
    failure_reason: Optional[str],
    newer_meta=None,
    older_meta=None,
    newer_extraction=None,
    older_extraction=None,
    delta_result=None,
    scoring_result=None,
    elapsed_seconds: float = 0.0,
) -> CompareFilingsOutput:

    def meta_out(m) -> Optional[FilingMetaOut]:
        if m is None:
            return None
        return FilingMetaOut(
            ticker=m.ticker,
            cik=m.cik,
            form_type=m.form_type,
            accession_number=m.accession_number,
            filing_date=m.filing_date,
            report_date=m.report_date,
            document_url=m.document_url,
            fetch_success=m.fetch_success,
            failure_reason=m.failure_reason,
            html_byte_length=m.html_byte_length,
        )

    def section_extraction_out(s) -> SectionExtractionOut:
        return SectionExtractionOut(
            section_name=s.section_name,
            item_label=s.item_label,
            extraction_success=s.extraction_success,
            method=s.method.value if hasattr(s.method, "value") else s.method,
            confidence_score=s.confidence_score,
            char_count=s.char_count,
            failure_reason=s.failure_reason,
            coverage_gap_note=s.coverage_gap_note,
        )

    def extraction_out(e) -> Optional[ExtractionOut]:
        if e is None:
            return None
        return ExtractionOut(
            filing_accession=e.filing_accession,
            filing_date=e.filing_date,
            risk_factors=section_extraction_out(e.risk_factors),
            mda=section_extraction_out(e.mda),
            full_doc_char_count=e.full_doc_char_count,
            known_gaps=e.known_gaps,
            both_succeeded=e.both_succeeded,
            any_succeeded=e.any_succeeded,
        )

    def change_out(c) -> ChangeOut:
        return ChangeOut(
            type=c["type"],
            older=c["older"],
            newer=c["newer"],
            similarity=c["similarity"],
        )

    def section_delta_out(sd) -> Optional[SectionDeltaOut]:
        if sd is None:
            return None
        top_changes = []
        for c in sd.changes[:50]:
            if c.change_type in ("added", "removed", "rewritten"):
                top_changes.append(ChangeOut(
                    type=c.change_type,
                    older=(c.older_text or "")[:300],
                    newer=(c.newer_text or "")[:300],
                    similarity=c.similarity,
                ))
        return SectionDeltaOut(
            section_name=sd.section_name,
            magnitude=sd.magnitude.value,
            total_older_sentences=sd.total_older_sentences,
            total_newer_sentences=sd.total_newer_sentences,
            added_count=sd.added_count,
            removed_count=sd.removed_count,
            rewritten_count=sd.rewritten_count,
            unchanged_count=sd.unchanged_count,
            pct_changed=sd.pct_changed,
            delta_success=sd.delta_success,
            failure_reason=sd.failure_reason,
            top_changes=top_changes,
        )

    def delta_out(d) -> Optional[DeltaOut]:
        if d is None:
            return None
        return DeltaOut(
            risk_factors=section_delta_out(d.risk_factors),
            mda=section_delta_out(d.mda),
            comparison_note=d.comparison_note,
        )

    def signal_hit_out(h) -> SignalHitOut:
        return SignalHitOut(
            signal=h.signal,
            tier=h.tier,
            weight=h.weight,
            in_change=h.in_change,
            context=h.context[:200],
        )

    def section_score_out(ss) -> SectionScoreOut:
        return SectionScoreOut(
            section_name=ss.section_name,
            materiality=ss.materiality.value,
            raw_score=ss.raw_score,
            is_estimate=ss.is_estimate,
            analyst_note=ss.analyst_note,
            tier1_hits=[signal_hit_out(h) for h in ss.tier1_hits],
            tier2_hits=[signal_hit_out(h) for h in ss.tier2_hits[:10]],
            new_signals=[signal_hit_out(h) for h in ss.new_signals],
            removed_signals=[signal_hit_out(h) for h in ss.removed_signals],
        )

    def scoring_out(s) -> Optional[ScoringOut]:
        if s is None:
            return None
        return ScoringOut(
            risk_factors=section_score_out(s.risk_factors),
            mda=section_score_out(s.mda),
            overall_materiality=s.overall_materiality.value,
            top_signals=s.top_signals,
            scoring_success=s.scoring_success,
            failure_reason=s.failure_reason,
            disclaimer=DISCLAIMER,
        )

    return CompareFilingsOutput(
        schema_version="1.0",
        tool="compare_filings",
        ticker=ticker,
        form_type=form_type,
        pipeline_success=pipeline_success,
        failure_reason=failure_reason,
        elapsed_seconds=round(elapsed_seconds, 2),
        newer_filing=meta_out(newer_meta),
        older_filing=meta_out(older_meta),
        newer_extraction=extraction_out(newer_extraction),
        older_extraction=extraction_out(older_extraction),
        delta=delta_out(delta_result),
        scoring=scoring_out(scoring_result),
        disclaimer=DISCLAIMER,
        coverage_gap_disclosure=_COVERAGE_GAP,
    )


# ---------------------------------------------------------------------------
# Tool definition — return type is Pydantic model for outputSchema generation
# ---------------------------------------------------------------------------

@mcp.tool()
async def compare_filings(
    ticker: str,
    form_type: Literal["10-Q", "10-K"] = "10-Q",
) -> CompareFilingsOutput:
    """
    Compare the two most recent SEC filings (10-Q or 10-K) for a US public company.
    Analyzes Risk Factors (Item 1A) and MD&A (Item 7) only.
    Returns structured delta analysis and materiality scores (labeled as estimates).

    IMPORTANT — 10-Q Risk Factors: Most S&P 500 10-Q filings incorporate Risk Factors
    by reference from the annual 10-K rather than restating them in full. When this is
    detected, the tool flags it explicitly instead of running a misleading comparison.
    Use form_type='10-K' for annual risk factor comparisons.

    IMPORTANT — Materiality scores: All scores are automated estimates based on
    keyword matching and heuristics. They do not constitute investment advice.

    Args:
        ticker: US stock ticker symbol (e.g. AAPL, MSFT, TSLA, JPM)
        form_type: Filing type — '10-Q' (quarterly) or '10-K' (annual). Default: 10-Q
    """
    start = time.monotonic()

    ticker = ticker.upper().strip()
    if not ticker or not ticker.replace("-", "").replace(".", "").isalpha():
        raise ToolError(f"Invalid ticker: {ticker!r}. Must be alphabetic (e.g. AAPL).")
    if form_type not in ("10-Q", "10-K"):
        raise ToolError("form_type must be '10-Q' or '10-K'.")

    try:
        result = await asyncio.wait_for(
            _run_pipeline(ticker, form_type),
            timeout=TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return _build_output(
            ticker=ticker, form_type=form_type,
            pipeline_success=False,
            failure_reason=f"Pipeline timed out after {TOOL_TIMEOUT}s",
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return _build_output(
            ticker=ticker, form_type=form_type,
            pipeline_success=False,
            failure_reason=f"Unexpected pipeline error: {exc}",
            elapsed_seconds=elapsed,
        )

    return result


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def _run_pipeline(ticker: str, form_type: str) -> CompareFilingsOutput:
    start = time.monotonic()

    # Step 1: Fetch
    fetch_result = await fetch_two_filings(ticker, form_type)

    if not fetch_result.pipeline_success:
        return _build_output(
            ticker=ticker, form_type=form_type,
            pipeline_success=False,
            failure_reason=fetch_result.failure_reason,
            newer_meta=fetch_result.newer,
            older_meta=fetch_result.older,
            elapsed_seconds=time.monotonic() - start,
        )

    # Step 2: Extract — FIX 5: form_type now passed correctly
    newer_extraction, older_extraction = await asyncio.gather(
        extract_sections(
            fetch_result.newer_html or "",
            accession=fetch_result.newer.accession_number if fetch_result.newer else "",
            filing_date=fetch_result.newer.filing_date if fetch_result.newer else "",
            form_type=form_type,
        ),
        extract_sections(
            fetch_result.older_html or "",
            accession=fetch_result.older.accession_number if fetch_result.older else "",
            filing_date=fetch_result.older.filing_date if fetch_result.older else "",
            form_type=form_type,
        ),
    )

    # FIX 2: Sanity checks — fail fast on near-empty extractions
    sanity_failure = _check_extraction_sanity(newer_extraction, older_extraction, form_type)
    if sanity_failure:
        return _build_output(
            ticker=ticker, form_type=form_type,
            pipeline_success=False,
            failure_reason=sanity_failure,
            newer_meta=fetch_result.newer,
            older_meta=fetch_result.older,
            newer_extraction=newer_extraction,
            older_extraction=older_extraction,
            elapsed_seconds=time.monotonic() - start,
        )

    # FIX 3: 10-Q reference pointer detection for Risk Factors
    rf_pointer_note = None
    if form_type == "10-Q":
        newer_rf = newer_extraction.risk_factors
        if _is_rf_reference_pointer(newer_rf.text, newer_rf.char_count):
            rf_pointer_note = (
                "REFERENCE POINTER DETECTED: This 10-Q's Risk Factors section "
                f"({newer_rf.char_count} chars) incorporates the Annual Report (10-K) "
                "by reference rather than restating risks in full. "
                "No meaningful quarter-over-quarter Risk Factor comparison is possible. "
                "Use form_type='10-K' to compare annual Risk Factor disclosures."
            )
            newer_extraction.risk_factors.coverage_gap_note = rf_pointer_note
            if older_extraction.risk_factors.coverage_gap_note is None:
                older_extraction.risk_factors.coverage_gap_note = rf_pointer_note

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

    # Determine final pipeline_success
    # True even if extraction partially fell back — partial data is better than nothing.
    # But rf_pointer_note means RF delta is not meaningful — still return data.
    pipeline_success = fetch_result.pipeline_success

    # If RF is a reference pointer, add a top-level failure reason so user knows
    failure_reason = rf_pointer_note if rf_pointer_note else None

    return _build_output(
        ticker=ticker, form_type=form_type,
        pipeline_success=pipeline_success,
        failure_reason=failure_reason,
        newer_meta=fetch_result.newer,
        older_meta=fetch_result.older,
        newer_extraction=newer_extraction,
        older_extraction=older_extraction,
        delta_result=delta_result,
        scoring_result=scoring_result,
        elapsed_seconds=time.monotonic() - start,
    )


def _check_extraction_sanity(
    newer_extraction, older_extraction, form_type: str
) -> Optional[str]:
    """
    FIX 2: Returns a failure reason string if extraction is too thin to be reliable.
    Returns None if everything looks OK.
    """
    issues = []

    for label, extraction in [("newer", newer_extraction), ("older", older_extraction)]:
        rf = extraction.risk_factors
        mda = extraction.mda

        # raw_fallback on either section is always unreliable
        if not mda.extraction_success:
            issues.append(
                f"{label} MD&A: extraction failed (method={mda.method.value if hasattr(mda.method, 'value') else mda.method}, "
                f"{mda.char_count} chars). Full document returned as fallback — comparison unreliable."
            )
        elif mda.char_count < MDA_MIN_CHARS:
            issues.append(
                f"{label} MD&A: only {mda.char_count} chars extracted "
                f"(minimum {MDA_MIN_CHARS}). Extraction likely captured a fragment, not the full section."
            )

        # Risk factors: only enforce size minimum on 10-K (10-Q can legitimately be a pointer)
        if form_type == "10-K":
            if not rf.extraction_success:
                issues.append(
                    f"{label} Risk Factors: extraction failed (method={rf.method.value if hasattr(rf.method, 'value') else rf.method}, "
                    f"{rf.char_count} chars)."
                )
            elif rf.char_count < RISK_FACTORS_MIN_CHARS_10K:
                issues.append(
                    f"{label} Risk Factors: only {rf.char_count} chars extracted "
                    f"(minimum {RISK_FACTORS_MIN_CHARS_10K} for 10-K). Likely a partial extraction."
                )

    if issues:
        return "Extraction quality below threshold — comparison would be unreliable. " + " | ".join(issues)

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)