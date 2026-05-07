"""
test_round2.py — Local verification against Alex's Round 2 test tickers
Run: python test_round2.py
"""
import asyncio
import sys

# Import pipeline directly — bypasses MCP layer
from server import _run_pipeline

TICKERS = [
    ("TSLA", "10-Q"),
    ("JPM",  "10-Q"),
    ("F",    "10-Q"),   # Ford — correct EDGAR ticker
    ("AAPL", "10-K"),
    ("NVDA", "10-K"),
]

async def test(ticker, form_type):
    print(f"\n{'='*60}")
    print(f"{ticker} {form_type}")
    print('='*60)
    try:
        r = await _run_pipeline(ticker, form_type)
        print(f"pipeline_success : {r.pipeline_success}")
        print(f"failure_reason   : {r.failure_reason}")
        print(f"elapsed_seconds  : {r.elapsed_seconds}s  {'✅ PASS' if r.elapsed_seconds < 35 else '❌ SLOW — over 35s'}")
        if r.newer_extraction:
            mda = r.newer_extraction.mda
            rf  = r.newer_extraction.risk_factors
            print(f"newer MD&A       : success={mda.extraction_success}, method={mda.method}, chars={mda.char_count}")
            print(f"newer RF         : success={rf.extraction_success},  method={rf.method}, chars={rf.char_count}")
            if rf.coverage_gap_note:
                print(f"RF pointer note  : {rf.coverage_gap_note[:120]}...")
        if r.scoring:
            print(f"overall_material : {r.scoring.overall_materiality}")
        else:
            print(f"scoring          : None (pipeline failed — correct)")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

async def main():
    for ticker, form_type in TICKERS:
        await test(ticker, form_type)
    print(f"\n{'='*60}")
    print("Done")

asyncio.run(main())