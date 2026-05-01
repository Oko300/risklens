import asyncio
from fetcher import fetch_two_filings
from extractor import extract_sections
from delta import compute_delta
from scorer import score_sections

async def test_ticker(ticker: str, form_type: str = "10-Q"):
    print(f"\nTesting {ticker}...")

    result = await fetch_two_filings(ticker, form_type=form_type)
    print(f"Pipeline success: {result.pipeline_success}")

    if not result.pipeline_success:
        print(f"Failure reason: {result.failure_reason}")
        return

    print(f"Newer filing: {result.newer.filing_date}")
    print(f"Older filing: {result.older.filing_date}")
    print(f"Newer URL: {result.newer.document_url}")

    current_sections = extract_sections(result.newer_html)
    previous_sections = extract_sections(result.older_html)

    print(f"Current extraction success: {current_sections['extraction_success']}")
    print(f"Previous extraction success: {previous_sections['extraction_success']}")
    print(f"Extraction method: {current_sections.get('extraction_method')}")

    delta = compute_delta(
        older_risk=previous_sections.get("risk_factors"),
        newer_risk=current_sections.get("risk_factors"),
        older_mda=previous_sections.get("mda"),
        newer_mda=current_sections.get("mda"),
    )

    print(f"\nRisk Factors Delta:")
    print(f"  Magnitude: {delta.risk_factors.magnitude}")
    print(f"  Added: {delta.risk_factors.added_count}")
    print(f"  Removed: {delta.risk_factors.removed_count}")
    print(f"  Rewritten: {delta.risk_factors.rewritten_count}")
    print(f"  Delta success: {delta.risk_factors.delta_success}")

    print(f"\nMDA Delta:")
    print(f"  Magnitude: {delta.mda.magnitude}")
    print(f"  Added: {delta.mda.added_count}")
    print(f"  Removed: {delta.mda.removed_count}")
    print(f"  Rewritten: {delta.mda.rewritten_count}")
    print(f"  Delta success: {delta.mda.delta_success}")

    scoring = score_sections(delta)
    print(f"\nScoring Result:")
    print(f"  Risk materiality: {scoring.risk_factors.materiality}")
    print(f"  MDA materiality: {scoring.mda.materiality}")
    print(f"  Disclaimer: {scoring.disclaimer}")

    print(f"\nComparison note: {delta.comparison_note}")

asyncio.run(test_ticker("PLTR"))