"""HEADERS = {
    "User-Agent": "RiskLens/1.0 (300joh@gmail.com)" 
fetcher.py — RiskLens EDGAR fetcher
Resolves ticker → CIK → two most recent 10-Q or 10-K filings → raw HTML.
Returns structured provenance at every step. Never fails silently.

Networking layer hardened for production:
- Shared AsyncClient with connection pooling (created once per process)
- Global rate limiter: max 2 concurrent SEC requests + 150 ms inter-request delay
- Retry with exponential backoff + jitter on all transient errors
- Deadline-aware: never starts a request if insufficient time remains
- In-memory submissions cache (TTL = 5 min) to avoid redundant SEC calls
- company_tickers.json cached for process lifetime
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse, unquote

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_FULL = "https://www.sec.gov"
HEADERS    = {
    "User-Agent":      "RiskLens/1.0 research-tool (contact@risklens.internal)",
    "Accept-Encoding": "gzip, deflate",
}

# Per-step timeouts (seconds) — tuned independently so no step starves another
TIMEOUT_CIK         = 20   # company_tickers.json (large, fetched once)
TIMEOUT_SUBMISSIONS = 15   # CIK submissions JSON
TIMEOUT_INDEX       = 20   # filing index HTML page
TIMEOUT_HTML        = 45   # primary document (can be very large)

# Global pipeline budget
PIPELINE_TIMEOUT    = 90.0  # seconds per ticker

# Retry policy
MAX_RETRIES         = 3
BACKOFF_BASE        = 1.0   # delay = BACKOFF_BASE * 2^(attempt-1) + jitter
BACKOFF_JITTER_MAX  = 0.5   # seconds of random jitter added per retry
MIN_REMAINING_S     = 3.0   # minimum seconds required before starting any request

# Rate limiting — SEC fair-use: max 2 concurrent, >=150 ms between requests
CONCURRENT_LIMIT    = 2
INTER_REQUEST_DELAY = 0.15  # seconds

ALLOWED_FORM_TYPES  = {"10-Q", "10-K"}

# Transient errors that warrant a retry (DNS failures surface as ConnectError)
_RETRYABLE = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
)

# ---------------------------------------------------------------------------
# Process-level shared state
# ---------------------------------------------------------------------------

# Single AsyncClient — created lazily, shared across all callers
_shared_client: Optional[httpx.AsyncClient] = None
_client_lock    = asyncio.Lock()

# Rate-limiting primitives — initialised lazily (needs running event loop)
_rate_semaphore: Optional[asyncio.Semaphore] = None
_rate_sem_lock   = asyncio.Lock()
_inter_req_lock  = asyncio.Lock()
_last_request_at: float = 0.0          # monotonic timestamp of last request end

# CIK cache
_tickers_cache: Optional[dict] = None
_tickers_lock   = asyncio.Lock()

# Submissions cache: cik -> (payload_dict, expiry_monotonic)
_submissions_cache: dict[str, tuple[dict, float]] = {}
_submissions_lock   = asyncio.Lock()
SUBMISSIONS_TTL     = 300.0            # 5-minute TTL


# ---------------------------------------------------------------------------
# Shared client — created once, reused forever
# ---------------------------------------------------------------------------

async def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        return _shared_client
    async with _client_lock:
        if _shared_client is None or _shared_client.is_closed:
            limits = httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30,
            )
            _shared_client = httpx.AsyncClient(
                headers=HEADERS,
                follow_redirects=True,
                timeout=httpx.Timeout(20.0),
                limits=limits,
            )
    return _shared_client


async def close_shared_client() -> None:
    """Call at process shutdown to cleanly drain the connection pool."""
    global _shared_client
    if _shared_client and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

async def _get_semaphore() -> asyncio.Semaphore:
    global _rate_semaphore
    if _rate_semaphore is not None:
        return _rate_semaphore
    async with _rate_sem_lock:
        if _rate_semaphore is None:
            _rate_semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    return _rate_semaphore


# ---------------------------------------------------------------------------
# Core fetch primitive — ALL HTTP calls go through here
# ---------------------------------------------------------------------------

async def fetch_with_retries(
    url: str,
    step_timeout: float,
    deadline: float,
) -> httpx.Response:
    """
    Rate-limited, deadline-aware GET with exponential backoff + jitter.

    Args:
        url:          Target URL.
        step_timeout: Maximum seconds for a single attempt.
        deadline:     Monotonic time by which the entire pipeline must finish.

    Returns:
        httpx.Response — caller must call .raise_for_status() if needed.

    Raises:
        TimeoutError:  Pipeline deadline exceeded.
        Last retryable exception after MAX_RETRIES attempts.
        Any non-retryable exception immediately.
    """
    global _last_request_at

    client    = await _get_client()
    semaphore = await _get_semaphore()
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, MAX_RETRIES + 1):
        # ── Deadline check ────────────────────────────────────────────────
        remaining = deadline - time.monotonic()
        if remaining < MIN_REMAINING_S:
            raise TimeoutError(
                f"Pipeline deadline reached before attempt {attempt} for {url} "
                f"({remaining:.1f}s remaining)"
            )

        # ── Effective timeout: never block past the deadline ──────────────
        effective_timeout = min(step_timeout, remaining - 0.5)
        if effective_timeout <= 0:
            raise TimeoutError(f"No time budget left for {url}")

        # ── Rate limiter: concurrency cap + inter-request spacing ─────────
        async with semaphore:
            # Enforce minimum gap between consecutive requests.
            # Lock is held only for the sleep calculation, not the fetch,
            # so other semaphore holders are not blocked during the HTTP call.
            async with _inter_req_lock:
                since_last = time.monotonic() - _last_request_at
                if since_last < INTER_REQUEST_DELAY:
                    await asyncio.sleep(INTER_REQUEST_DELAY - since_last)

            try:
                resp             = await client.get(url, timeout=effective_timeout)
                _last_request_at = time.monotonic()
                return resp

            except _RETRYABLE as exc:
                _last_request_at = time.monotonic()
                last_exc         = exc

            except Exception:
                _last_request_at = time.monotonic()
                raise   # Non-retryable — propagate immediately

        # ── Backoff (only reached on retryable failure inside semaphore) ──
        if attempt == MAX_RETRIES:
            break

        # Exponential backoff: 1s, 2s, 4s … plus random jitter
        jitter  = random.uniform(0, BACKOFF_JITTER_MAX)
        backoff = BACKOFF_BASE * (2 ** (attempt - 1)) + jitter
        wait    = min(backoff, deadline - time.monotonic() - MIN_REMAINING_S)
        if wait > 0:
            await asyncio.sleep(wait)

    raise last_exc   # All retries exhausted


async def _get_json(url: str, step_timeout: float, deadline: float) -> dict:
    """fetch_with_retries -> raise_for_status -> parse JSON."""
    resp = await fetch_with_retries(url, step_timeout, deadline)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilingMeta:
    ticker: str
    cik: str
    form_type: str
    accession_number: str
    filing_date: str
    report_date: str
    document_url: str
    fetch_timestamp: float = field(default_factory=time.time)
    html_byte_length: int = 0
    fetch_success: bool = False
    failure_reason: Optional[str] = None


@dataclass
class FetcherResult:
    ticker: str
    cik: str
    form_type: str
    newer: Optional[FilingMeta]
    older: Optional[FilingMeta]
    newer_html: Optional[str]
    older_html: Optional[str]
    pipeline_success: bool
    failure_reason: Optional[str] = None
    coverage_gap_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_two_filings(
    ticker: str,
    form_type: Literal["10-Q", "10-K"] = "10-Q",
    timeout_total: float = PIPELINE_TIMEOUT,
) -> FetcherResult:
    if form_type not in ALLOWED_FORM_TYPES:
        return _fail(ticker, "", form_type, f"form_type must be one of {ALLOWED_FORM_TYPES}")

    ticker   = ticker.upper().strip()
    deadline = time.monotonic() + timeout_total

    # Step 1: ticker -> CIK
    try:
        cik = await _resolve_cik(ticker, deadline)
    except Exception as exc:
        return _fail(ticker, "", form_type,
                     f"CIK resolution failed: {type(exc).__name__}: {exc}")

    # Step 2: CIK -> filing list (with URL resolution)
    try:
        filings = await _get_filing_list(cik, form_type, deadline)
    except Exception as exc:
        return _fail(ticker, cik, form_type,
                     f"Submissions fetch failed: {type(exc).__name__}: {exc}")

    if len(filings) < 2:
        gap = (
            f"Only {len(filings)} {form_type} filing(s) found for {ticker} "
            f"(CIK {cik}). Comparison requires at least two."
        )
        return FetcherResult(
            ticker=ticker, cik=cik, form_type=form_type,
            newer=filings[0] if filings else None, older=None,
            newer_html=None, older_html=None,
            pipeline_success=False,
            failure_reason=gap, coverage_gap_note=gap,
        )

    newer_meta, older_meta = filings[0], filings[1]

    # Step 3: fetch HTML for both filings — failures are isolated, never fatal
    newer_html, newer_meta = await _fetch_filing_html(newer_meta, deadline)
    older_html, older_meta = await _fetch_filing_html(older_meta, deadline)

    ok       = newer_meta.fetch_success and older_meta.fetch_success
    fail_msg = None
    if not ok:
        parts = []
        if not newer_meta.fetch_success:
            parts.append(f"newer HTML: {newer_meta.failure_reason}")
        if not older_meta.fetch_success:
            parts.append(f"older HTML: {older_meta.failure_reason}")
        fail_msg = "; ".join(parts)

    return FetcherResult(
        ticker=ticker, cik=cik, form_type=form_type,
        newer=newer_meta, older=older_meta,
        newer_html=newer_html, older_html=older_html,
        pipeline_success=ok, failure_reason=fail_msg,
    )


# ---------------------------------------------------------------------------
# CIK resolution — process-lifetime cache
# ---------------------------------------------------------------------------

async def _resolve_cik(ticker: str, deadline: float) -> str:
    global _tickers_cache

    async with _tickers_lock:
        if _tickers_cache is None:
            if deadline - time.monotonic() < MIN_REMAINING_S:
                raise TimeoutError("No time remaining for CIK resolution")
            url            = f"{EDGAR_FULL}/files/company_tickers.json"
            _tickers_cache = await _get_json(url, TIMEOUT_CIK, deadline)

    for entry in _tickers_cache.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company_tickers.json")


# ---------------------------------------------------------------------------
# Filing list — with per-CIK submissions cache
# ---------------------------------------------------------------------------

async def _get_submissions(cik: str, deadline: float) -> dict:
    """Fetch submissions JSON for a CIK, backed by a short-lived in-memory cache."""
    now = time.monotonic()

    async with _submissions_lock:
        cached = _submissions_cache.get(cik)
        if cached and now < cached[1]:
            return cached[0]

    url  = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    data = await _get_json(url, TIMEOUT_SUBMISSIONS, deadline)

    async with _submissions_lock:
        _submissions_cache[cik] = (data, time.monotonic() + SUBMISSIONS_TTL)

    return data


async def _get_filing_list(
    cik: str, form_type: str, deadline: float,
) -> list[FilingMeta]:
    data        = await _get_submissions(cik, deadline)
    all_filings = _extract_filings_from_submissions(data, form_type)

    # Fetch paginated filing archives only if we still need more entries
    for extra in data.get("filings", {}).get("files", []):
        if len(all_filings) >= 2:
            break
        if deadline - time.monotonic() < MIN_REMAINING_S:
            break
        try:
            name      = extra["name"]
            extra_url = (
                f"{EDGAR_BASE}{name}" if name.startswith("/")
                else f"{EDGAR_BASE}/submissions/{name}"
            )
            extra_data = await _get_json(extra_url, TIMEOUT_SUBMISSIONS, deadline)
            all_filings.extend(_extract_filings_from_submissions(extra_data, form_type))
        except Exception:
            pass   # Non-fatal: continue with whatever we have

    ticker_name = data.get("tickers", [""])[0] or cik
    results: list[FilingMeta] = []

    for f in all_filings[:2]:
        if deadline - time.monotonic() < MIN_REMAINING_S:
            results.append(FilingMeta(
                ticker=ticker_name, cik=cik, form_type=form_type,
                accession_number=f["accession"],
                filing_date=f["filing_date"], report_date=f["report_date"],
                document_url="", fetch_success=False,
                failure_reason="Deadline reached before document URL lookup",
            ))
            continue
        try:
            doc_url = await _resolve_primary_document(cik, f["accession"], deadline)
            results.append(FilingMeta(
                ticker=ticker_name, cik=cik, form_type=form_type,
                accession_number=f["accession"],
                filing_date=f["filing_date"], report_date=f["report_date"],
                document_url=doc_url,
            ))
        except Exception as exc:
            results.append(FilingMeta(
                ticker=ticker_name, cik=cik, form_type=form_type,
                accession_number=f["accession"],
                filing_date=f["filing_date"], report_date=f["report_date"],
                document_url="", fetch_success=False,
                failure_reason=(
                    f"Document URL resolution failed: {type(exc).__name__}: {exc}"
                ),
            ))

    return results


def _extract_filings_from_submissions(data: dict, form_type: str) -> list[dict]:
    recent       = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    results      = []
    for i, form in enumerate(forms):
        if form == form_type:
            results.append({
                "accession":   accessions[i].replace("-", ""),
                "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                "report_date": report_dates[i] if i < len(report_dates) else "",
            })
    return results


# ---------------------------------------------------------------------------
# Document URL resolution
# ---------------------------------------------------------------------------

async def _resolve_primary_document(
    cik: str, accession_no_dashes: str, deadline: float,
) -> str:
    acc_dashes = (
        f"{accession_no_dashes[:10]}-"
        f"{accession_no_dashes[10:12]}-"
        f"{accession_no_dashes[12:]}"
    )
    index_url = (
        f"{EDGAR_FULL}/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/"
        f"{acc_dashes}-index.htm"
    )

    resp = await fetch_with_retries(index_url, TIMEOUT_INDEX, deadline)
    resp.raise_for_status()

    from bs4 import BeautifulSoup
    soup     = BeautifulSoup(resp.text, "lxml")
    best_url = None

    # Strategy 1: table row where type column is an allowed form type
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type in ALLOWED_FORM_TYPES:
                link = cells[2].find("a")
                if link and link.get("href"):
                    best_url = link["href"]
                    break

    # Strategy 2: first .htm link that is not the index itself
    if not best_url:
        for a in soup.select("table a[href]"):
            href = a["href"]
            if href.endswith((".htm", ".html")) and "index" not in href.lower():
                best_url = href
                break

    if not best_url:
        raise ValueError(f"No primary document found in index: {index_url}")

    if best_url.startswith("http"):
        full_url = best_url
    elif best_url.startswith("/"):
        full_url = f"{EDGAR_FULL}{best_url}"
    else:
        full_url = (
            f"{EDGAR_FULL}/Archives/edgar/data/{int(cik)}/"
            f"{accession_no_dashes}/{best_url}"
        )

    return _unwrap_ix_viewer_url(full_url)


def _unwrap_ix_viewer_url(url: str) -> str:
    """Strip EDGAR iXBRL viewer wrapper: /ix?doc=/Archives/... -> direct file URL."""
    if "/ix?doc=" in url:
        doc_path = parse_qs(urlparse(url).query).get("doc", [""])[0]
        if doc_path:
            doc_path = unquote(doc_path)
            return f"{EDGAR_FULL}{doc_path}" if doc_path.startswith("/") else doc_path
    return url


# ---------------------------------------------------------------------------
# HTML fetch — isolated; failure populates FilingMeta, does not raise
# ---------------------------------------------------------------------------

async def _fetch_filing_html(
    meta: FilingMeta, deadline: float,
) -> tuple[Optional[str], FilingMeta]:
    if not meta.document_url:
        meta.fetch_success  = False
        meta.failure_reason = meta.failure_reason or "No document URL available"
        return None, meta

    if deadline - time.monotonic() < MIN_REMAINING_S:
        meta.fetch_success  = False
        meta.failure_reason = "Pipeline timeout before HTML fetch"
        return None, meta

    try:
        resp = await fetch_with_retries(meta.document_url, TIMEOUT_HTML, deadline)
        resp.raise_for_status()
        html                  = resp.text
        meta.html_byte_length = len(html.encode("utf-8"))
        meta.fetch_success    = True
        return html, meta
    except httpx.HTTPStatusError as exc:
        meta.fetch_success  = False
        meta.failure_reason = f"HTTP {exc.response.status_code} at {meta.document_url}"
        return None, meta
    except Exception as exc:
        meta.fetch_success  = False
        meta.failure_reason = f"{type(exc).__name__}: {exc}"
        return None, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(ticker: str, cik: str, form_type: str, reason: str) -> FetcherResult:
    return FetcherResult(
        ticker=ticker, cik=cik, form_type=form_type,
        newer=None, older=None, newer_html=None, older_html=None,
        pipeline_success=False, failure_reason=reason,
    )


def reset_tickers_cache() -> None:
    """Reset the CIK cache — useful for testing."""
    global _tickers_cache
    _tickers_cache = None


def reset_submissions_cache() -> None:
    """Flush the submissions cache — useful for testing or forced refresh."""
    global _submissions_cache
    _submissions_cache = {}