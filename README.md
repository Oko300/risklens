# RiskLens

Compares the two most recent 10-Q or 10-K filings for a US public company,
focusing exclusively on **Risk Factors (Item 1A)** and **MD&A (Item 7)**.

> **Scope**: Filing comparison only. No monitoring, alerts, or dashboards.  
> **Coverage**: 10-Q and 10-K only. US domestic filers only (CIK-resolvable tickers).  
> **Outputs**: All materiality assessments are labeled as **ESTIMATES**.

---

## Setup

```bash
pip install fastmcp httpx beautifulsoup4 lxml python-dotenv pytest
cp .env.example .env   # Add CTX_PUBLIC_KEY if using ctxprotocol auth
```

## Run server

```bash
python server.py
```

## Run unit tests (no network required)

```bash
pytest test_unit.py -v
```

## Run EDGAR integration tests (network required)

```bash
# Test all 50+ tickers (10-Q)
python test_pipeline.py --report results.json

# Test specific tickers
python test_pipeline.py --tickers AAPL MSFT TSLA --form 10-Q

# Test 10-K filings
python test_pipeline.py --form 10-K
```

---

## Architecture

```
fetcher.py    → ticker → CIK → two most recent filings → HTML
extractor.py  → HTML → Risk Factors text + MD&A text (per-section provenance)
delta.py      → sentence-level diff → change magnitude classification
scorer.py     → keyword signal detection → materiality labels (ESTIMATES)
server.py     → FastMCP tool with ctxprotocol JWT auth + explicit output schema
```

### Extraction strategy (in priority order)

1. **Anchor/ID targeting** — `<a name="item1a">` or `id="item1a"` (confidence: 0.92)
2. **Heading tag traversal** — h1–h4, bold, strong matching Item patterns (confidence: 0.85)
3. **Table of contents links** — follow `href="#..."` TOC links (confidence: 0.78)
4. **Pattern match on full text** — regex on normalized plain text (confidence: 0.70)
5. **Raw fallback** — full document text returned with `extraction_success=False` flag

Each section tracks its own extraction method and confidence independently.

---

## Output schema (v1.0)

```json
{
  "schema_version": "1.0",
  "ticker": "AAPL",
  "form_type": "10-Q",
  "pipeline_success": true,
  "newer_filing": { "filing_date": "...", "fetch_success": true, ... },
  "older_filing": { "filing_date": "...", "fetch_success": true, ... },
  "newer_extraction": {
    "risk_factors": { "extraction_success": true, "method": "heading_tag", "confidence_score": 0.85, ... },
    "mda":          { "extraction_success": true, "method": "anchor_href", "confidence_score": 0.92, ... },
    "known_gaps": []
  },
  "delta": {
    "risk_factors": { "magnitude": "moderate", "pct_changed": 0.18, "added_count": 3, ... },
    "mda":          { "magnitude": "major",    "pct_changed": 0.41, ... },
    "comparison_note": "ESTIMATE: ..."
  },
  "scoring": {
    "overall_materiality": "high",
    "top_signals": ["going concern [risk_factors]", ...],
    "disclaimer": "ESTIMATE — All materiality assessments ...",
    ...
  },
  "disclaimer": "ESTIMATE — ...",
  "coverage_gap_disclosure": "RiskLens only analyzes 10-Q and 10-K ..."
}
```

---

## Known coverage gaps

- Foreign private issuers (20-F) are **not supported**
- Exhibit-only or amendment filings may not contain full section text
- Image-based PDFs filed as HTML will produce raw-fallback extraction
- XBRL inline filings with unusual tag structure may fall back to pattern matching
- Companies with fewer than 2 filings of the requested type cannot be compared

---

## Grant compliance

| Requirement | Status |
|---|---|
| Only 10-Q and 10-K | ✓ |
| Only Risk Factors and MD&A | ✓ |
| Only two most recent filings | ✓ |
| No alerts / monitoring / dashboards | ✓ |
| Interpretive outputs labeled as estimates | ✓ |
| Known coverage gaps disclosed | ✓ |
| Silent failures → raw fallback + failure flag | ✓ |
| Test ≥50 filings before review | Run `test_pipeline.py` |
| Respond within 60 seconds | Enforced via `asyncio.wait_for` |