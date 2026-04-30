"""
test_pipeline.py — RiskLens filing test harness
Runs the full pipeline against real EDGAR filings and produces a review report.
Required: test at least 50 filings before Context submission.

Usage:
    python test_pipeline.py                        # Run all tickers
    python test_pipeline.py --tickers AAPL MSFT    # Run specific tickers
    python test_pipeline.py --form 10-K            # Test 10-Ks only
    python test_pipeline.py --report results.json  # Save JSON report
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 50+ ticker test list — diversified by sector, size, filing complexity
# ---------------------------------------------------------------------------

TEST_TICKERS_10Q = [
    # Mega-cap tech (complex filings, inline XBRL)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Large-cap tech / software
    "CRM", "ADBE", "ORCL", "INTC", "AMD", "QCOM", "TXN",
    # Financials (complex risk disclosures)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B",
    # Healthcare / pharma
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD",
    # Consumer / retail
    "WMT", "HD", "TGT", "COST", "MCD", "SBUX", "NKE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "MPC",
    # Industrials
    "BA", "CAT", "GE", "HON", "MMM", "LMT", "RTX",
    # Telecom / media
    "T", "VZ", "CMCSA", "DIS", "NFLX",
    # Smaller / growth (more likely to have going-concern or unusual risk language)
    "SNAP", "LYFT", "RIVN", "LCID", "HOOD",
    # REITs (very different MD&A structure)
    "PLD", "AMT", "O",
]

TEST_TICKERS_10K = [
    "AAPL", "MSFT", "AMZN", "JPM", "XOM",
    "JNJ", "WMT", "BA", "NFLX", "RIVN",
]


@dataclass
class FilingTestResult:
    ticker: str
    form_type: str
    pipeline_success: bool
    newer_date: Optional[str]
    older_date: Optional[str]
    risk_extraction_method: Optional[str]
    risk_extraction_success: Optional[bool]
    mda_extraction_method: Optional[str]
    mda_extraction_success: Optional[bool]
    risk_magnitude: Optional[str]
    mda_magnitude: Optional[str]
    overall_materiality: Optional[str]
    top_signals: list[str]
    known_gaps: list[str]
    failure_reason: Optional[str]
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class TestReport:
    run_timestamp: str
    form_type_tested: str
    total_tickers: int
    total_attempted: int
    pipeline_success_count: int
    extraction_both_success: int
    extraction_partial_success: int
    extraction_complete_failure: int
    timeout_count: int
    results: list[FilingTestResult] = field(default_factory=list)
    summary_notes: list[str] = field(default_factory=list)

    @property
    def pipeline_success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.pipeline_success_count / self.total_attempted

    @property
    def extraction_success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.extraction_both_success / self.total_attempted


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_one(ticker: str, form_type: str) -> FilingTestResult:
    """Run the full pipeline for one ticker and capture structured results."""
    start = time.monotonic()

    try:
        # Import here to allow running tests without server.py's MCP layer
        from fetcher import fetch_two_filings
        from extractor import extract_sections
        from delta import compute_delta
        from scorer import score_sections

        fetch = await fetch_two_filings(ticker, form_type)
        elapsed_fetch = time.monotonic() - start

        if not fetch.pipeline_success:
            return FilingTestResult(
                ticker=ticker, form_type=form_type,
                pipeline_success=False,
                newer_date=fetch.newer.filing_date if fetch.newer else None,
                older_date=fetch.older.filing_date if fetch.older else None,
                risk_extraction_method=None, risk_extraction_success=None,
                mda_extraction_method=None, mda_extraction_success=None,
                risk_magnitude=None, mda_magnitude=None,
                overall_materiality=None, top_signals=[],
                known_gaps=[fetch.coverage_gap_note] if fetch.coverage_gap_note else [],
                failure_reason=fetch.failure_reason,
                elapsed_seconds=round(time.monotonic() - start, 2),
            )

        newer_ext = extract_sections(
            fetch.newer_html or "",
            accession=fetch.newer.accession_number if fetch.newer else "",
            filing_date=fetch.newer.filing_date if fetch.newer else "",
            form_type=form_type,
        )
        older_ext = extract_sections(
            fetch.older_html or "",
            accession=fetch.older.accession_number if fetch.older else "",
            filing_date=fetch.older.filing_date if fetch.older else "",
            form_type=form_type,
        )

        delta = compute_delta(
            older_risk=older_ext.risk_factors.text,
            newer_risk=newer_ext.risk_factors.text,
            older_mda=older_ext.mda.text,
            newer_mda=newer_ext.mda.text,
        )

        scoring = score_sections(
            newer_risk_text=newer_ext.risk_factors.text,
            older_risk_text=older_ext.risk_factors.text,
            newer_mda_text=newer_ext.mda.text,
            older_mda_text=older_ext.mda.text,
            risk_delta=delta.risk_factors,
            mda_delta=delta.mda,
        )

        gaps = newer_ext.known_gaps + older_ext.known_gaps

        return FilingTestResult(
            ticker=ticker, form_type=form_type,
            pipeline_success=True,
            newer_date=fetch.newer.filing_date if fetch.newer else None,
            older_date=fetch.older.filing_date if fetch.older else None,
            risk_extraction_method=newer_ext.risk_factors.method.value,
            risk_extraction_success=newer_ext.risk_factors.extraction_success,
            mda_extraction_method=newer_ext.mda.method.value,
            mda_extraction_success=newer_ext.mda.extraction_success,
            risk_magnitude=delta.risk_factors.magnitude.value,
            mda_magnitude=delta.mda.magnitude.value,
            overall_materiality=scoring.overall_materiality.value,
            top_signals=scoring.top_signals[:5],
            known_gaps=gaps[:5],
            failure_reason=None,
            elapsed_seconds=round(time.monotonic() - start, 2),
        )

    except asyncio.TimeoutError:
        return FilingTestResult(
            ticker=ticker, form_type=form_type, pipeline_success=False,
            newer_date=None, older_date=None,
            risk_extraction_method=None, risk_extraction_success=None,
            mda_extraction_method=None, mda_extraction_success=None,
            risk_magnitude=None, mda_magnitude=None,
            overall_materiality=None, top_signals=[], known_gaps=[],
            failure_reason="Timeout",
            elapsed_seconds=round(time.monotonic() - start, 2),
            error="TimeoutError",
        )
    except Exception as exc:
        return FilingTestResult(
            ticker=ticker, form_type=form_type, pipeline_success=False,
            newer_date=None, older_date=None,
            risk_extraction_method=None, risk_extraction_success=None,
            mda_extraction_method=None, mda_extraction_success=None,
            risk_magnitude=None, mda_magnitude=None,
            overall_materiality=None, top_signals=[], known_gaps=[],
            failure_reason=str(exc),
            elapsed_seconds=round(time.monotonic() - start, 2),
            error=type(exc).__name__,
        )


async def run_all(tickers: list[str], form_type: str, concurrency: int = 3) -> TestReport:
    """
    Run tests for all tickers with controlled concurrency.
    EDGAR rate limits: keep concurrency low (2-3 max).
    """
    from datetime import datetime, timezone

    report = TestReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        form_type_tested=form_type,
        total_tickers=len(tickers),
        total_attempted=0,
        pipeline_success_count=0,
        extraction_both_success=0,
        extraction_partial_success=0,
        extraction_complete_failure=0,
        timeout_count=0,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_sem(ticker):
        async with semaphore:
            result = await run_one(ticker, form_type)
            # Small delay to be polite to EDGAR
            await asyncio.sleep(0.5)
            return result

    print(f"[RiskLens Test] Running {len(tickers)} tickers ({form_type}) with concurrency={concurrency}")
    print("-" * 70)

    tasks = [run_with_sem(t) for t in tickers]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result: FilingTestResult = await coro
        report.results.append(result)
        report.total_attempted += 1

        # Tally
        if result.pipeline_success:
            report.pipeline_success_count += 1
            if result.risk_extraction_success and result.mda_extraction_success:
                report.extraction_both_success += 1
            elif result.risk_extraction_success or result.mda_extraction_success:
                report.extraction_partial_success += 1
            else:
                report.extraction_complete_failure += 1
        else:
            if result.error == "TimeoutError" or result.failure_reason == "Timeout":
                report.timeout_count += 1

        # Live progress
        status = "✓" if result.pipeline_success else "✗"
        ext_status = ""
        if result.pipeline_success:
            r = "R✓" if result.risk_extraction_success else "R✗(fallback)"
            m = "M✓" if result.mda_extraction_success else "M✗(fallback)"
            ext_status = f"  {r} {m}  {result.risk_magnitude}/{result.mda_magnitude}  {result.overall_materiality}"
        else:
            ext_status = f"  FAIL: {(result.failure_reason or '')[:60]}"

        print(f"[{i+1:3d}/{len(tickers)}] {status} {result.ticker:8s} ({result.elapsed_seconds:.1f}s){ext_status}")

    # Summary notes
    report.summary_notes = _build_summary_notes(report)
    return report


def _build_summary_notes(report: TestReport) -> list[str]:
    notes = []
    notes.append(
        f"Pipeline success rate: {report.pipeline_success_rate:.1%} "
        f"({report.pipeline_success_count}/{report.total_attempted})"
    )
    notes.append(
        f"Full extraction success: {report.extraction_success_rate:.1%} "
        f"({report.extraction_both_success}/{report.total_attempted})"
    )
    notes.append(
        f"Partial extraction (one section failed): {report.extraction_partial_success}"
    )
    notes.append(
        f"Complete extraction failure (raw fallback): {report.extraction_complete_failure}"
    )
    notes.append(f"Timeouts: {report.timeout_count}")

    # Method breakdown
    from collections import Counter
    risk_methods = Counter(
        r.risk_extraction_method for r in report.results
        if r.risk_extraction_method
    )
    mda_methods = Counter(
        r.mda_extraction_method for r in report.results
        if r.mda_extraction_method
    )
    notes.append(f"Risk extraction methods: {dict(risk_methods)}")
    notes.append(f"MD&A extraction methods: {dict(mda_methods)}")

    # Failures
    failures = [r for r in report.results if not r.pipeline_success]
    if failures:
        fail_reasons = Counter(r.failure_reason for r in failures)
        notes.append(f"Failure reasons: {dict(fail_reasons)}")

    return notes


def _print_report(report: TestReport):
    print("\n" + "=" * 70)
    print("RISKLENS TEST REPORT")
    print("=" * 70)
    for note in report.summary_notes:
        print(f"  {note}")
    print()

    # Flag tickers that fell back to raw text
    fallbacks = [
        r for r in report.results
        if r.pipeline_success and (
            not r.risk_extraction_success or not r.mda_extraction_success
        )
    ]
    if fallbacks:
        print(f"TICKERS WITH EXTRACTION FALLBACKS ({len(fallbacks)}):")
        for r in fallbacks:
            parts = []
            if not r.risk_extraction_success:
                parts.append(f"Risk({r.risk_extraction_method})")
            if not r.mda_extraction_success:
                parts.append(f"MDA({r.mda_extraction_method})")
            print(f"  {r.ticker}: {', '.join(parts)}")
        print()

    # Failures
    failures = [r for r in report.results if not r.pipeline_success]
    if failures:
        print(f"PIPELINE FAILURES ({len(failures)}):")
        for r in failures:
            print(f"  {r.ticker}: {r.failure_reason}")
        print()

    # Grant readiness check
    print("GRANT READINESS:")
    n = report.total_attempted
    if n >= 50:
        print(f"  ✓ Tested {n} filings (grant requires 50)")
    else:
        print(f"  ✗ Only tested {n} filings (grant requires 50 — add more tickers)")

    if report.pipeline_success_rate >= 0.85:
        print(f"  ✓ Pipeline success rate {report.pipeline_success_rate:.1%} (target ≥85%)")
    else:
        print(f"  ✗ Pipeline success rate {report.pipeline_success_rate:.1%} (target ≥85%)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="RiskLens filing test harness")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to test")
    parser.add_argument("--form", choices=["10-Q", "10-K"], default="10-Q")
    parser.add_argument("--report", help="Save JSON report to file")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.form == "10-K":
        tickers = TEST_TICKERS_10K
    else:
        tickers = TEST_TICKERS_10Q

    report = await run_all(tickers, args.form, concurrency=args.concurrency)
    _print_report(report)

    if args.report:
        with open(args.report, "w") as f:
            # Convert dataclasses for JSON serialization
            report_dict = {
                "run_timestamp": report.run_timestamp,
                "form_type_tested": report.form_type_tested,
                "total_tickers": report.total_tickers,
                "total_attempted": report.total_attempted,
                "pipeline_success_count": report.pipeline_success_count,
                "pipeline_success_rate": report.pipeline_success_rate,
                "extraction_both_success": report.extraction_both_success,
                "extraction_success_rate": report.extraction_success_rate,
                "extraction_partial_success": report.extraction_partial_success,
                "extraction_complete_failure": report.extraction_complete_failure,
                "timeout_count": report.timeout_count,
                "summary_notes": report.summary_notes,
                "results": [
                    {
                        "ticker": r.ticker,
                        "form_type": r.form_type,
                        "pipeline_success": r.pipeline_success,
                        "newer_date": r.newer_date,
                        "older_date": r.older_date,
                        "risk_extraction_method": r.risk_extraction_method,
                        "risk_extraction_success": r.risk_extraction_success,
                        "mda_extraction_method": r.mda_extraction_method,
                        "mda_extraction_success": r.mda_extraction_success,
                        "risk_magnitude": r.risk_magnitude,
                        "mda_magnitude": r.mda_magnitude,
                        "overall_materiality": r.overall_materiality,
                        "top_signals": r.top_signals,
                        "known_gaps": r.known_gaps,
                        "failure_reason": r.failure_reason,
                        "elapsed_seconds": r.elapsed_seconds,
                    }
                    for r in sorted(report.results, key=lambda x: x.ticker)
                ],
            }
            json.dump(report_dict, f, indent=2)
        print(f"[RiskLens Test] Report saved to {args.report}")

    # Exit with non-zero if below 50 filings or below 85% success
    if report.total_attempted < 50 or report.pipeline_success_rate < 0.85:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())