"""
extractor.py — RiskLens section extractor
Extracts Risk Factors and MD&A from 10-Q / 10-K HTML.
Tracks extraction method and confidence independently per section.
Never fails silently: always returns raw-text fallback + failure flag.

KEY FIXES:
1. 10-Q MD&A = Item 2 / 10-K MD&A = Item 7 (form_type aware)
2. TOC entry detection: if collected text is too short, skip and find next match
   (EDGAR iXBRL TOC has "Item 2. MD&A" entries right above "Item 3" — 
    the extractor was stopping immediately after the TOC hit)
3. Cross-reference guard: "See Item 3" inside running text does not stop collection
   (only standalone bold headings stop collection)
4. 10-K MD&A start patterns require "management" or "discussion" in heading —
   prevents confusing "Item 7 — Reserved" or other short Item 7 entries
"""

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Minimum chars for extracted text to be considered a real section (not a TOC hit)
MIN_SECTION_CHARS = 500


class ExtractionMethod(str, Enum):
    HEADING_TAG       = "heading_tag"
    IXBRL_DIV         = "ixbrl_div"
    PATTERN_MATCH     = "pattern_match"
    ANCHOR_HREF       = "anchor_href"
    TABLE_OF_CONTENTS = "toc_link"
    RAW_FALLBACK      = "raw_fallback"
    FAILED            = "failed"


@dataclass
class SectionResult:
    section_name: str
    item_label: str
    text: Optional[str]
    method: ExtractionMethod
    extraction_success: bool
    failure_reason: Optional[str] = None
    char_count: int = 0
    confidence_score: float = 0.0
    coverage_gap_note: Optional[str] = None

    def __post_init__(self):
        if self.text:
            self.char_count = len(self.text)


@dataclass
class ExtractionResult:
    filing_accession: str
    filing_date: str
    form_type: str
    risk_factors: SectionResult
    mda: SectionResult
    full_doc_char_count: int = 0
    known_gaps: list[str] = field(default_factory=list)

    @property
    def both_succeeded(self) -> bool:
        return self.risk_factors.extraction_success and self.mda.extraction_success

    @property
    def any_succeeded(self) -> bool:
        return self.risk_factors.extraction_success or self.mda.extraction_success


# ---------------------------------------------------------------------------
# Section specs — form-type aware
# ---------------------------------------------------------------------------
#
# 10-Q structure:              10-K structure:
#   Item 1  Financial Stmts     Item 1   Business
#   Item 1A Risk Factors        Item 1A  Risk Factors
#   Item 1B Unresolved Staff    Item 1B  Unresolved Staff
#   Item 2  MD&A          <--   Item 2   Properties  (NOT MD&A)
#   Item 3  Quant/Qual          Item 7   MD&A         <--
#   Item 4  Controls            Item 7A  Quant/Qual
#                               Item 8   Financial Stmts

def _build_specs(form_type: str) -> dict:
    risk_spec = {
        "item_label": "Item 1A",
        "display":    "Risk Factors",
        "start_patterns": [
            r"item\s+1a[\.\s\u2014\-\u2013]+risk\s+factors",
            r"item\s+1a\b",
        ],
        "end_patterns": [
            r"item\s+1b\b",
            r"item\s+2\b",
        ],
        "next_items": ["item 1b", "item 2"],
        "keywords": ["risk", "could", "may", "uncertain", "adverse", "material",
                     "competition", "regulatory", "harm", "impact", "exposure",
                     "liability", "loss", "failure", "breach"],
    }

    if form_type == "10-Q":
        mda_spec = {
            "item_label": "Item 2",
            "display":    "Management's Discussion and Analysis",
            # Require "management" or "discussion" in the heading — prevents grabbing
            # "Item 2 — Unregistered Sales" which also uses bold Item 2 style
            "start_patterns": [
                r"item\s+2[\.\s\u2014\-\u2013]+management",
                r"item\s+2[\.\s]+discussion",
                r"item\s+2[\.\s]+md",
            ],
            "end_patterns": [
                r"item\s+3\b",
                r"item\s+3[\.\s]+quantitative",
                r"item\s+4\b",
            ],
            "next_items": ["item 3", "item 4"],
            "keywords": ["revenue", "operating", "liquidity", "results", "cash",
                         "quarter", "year", "income", "loss", "expense", "financial",
                         "increased", "decreased", "compared", "net"],
        }
    else:  # 10-K
        mda_spec = {
            "item_label": "Item 7",
            "display":    "Management's Discussion and Analysis",
            # ONLY match Item 7 with "management" or "discussion" in heading.
            # "item\s+7\b" alone would also match "Item 7A" or "Item 7 — Reserved"
            "start_patterns": [
                r"item\s+7[\.\s\u2014\-\u2013]+management",
                r"item\s+7[\.\s]+discussion",
                r"item\s+7[\.\s]+md",
            ],
            "end_patterns": [
                r"item\s+7a\b",
                r"item\s+8\b",
                r"item\s+8[\.\s]+financial",
            ],
            "next_items": ["item 7a", "item 8"],
            "keywords": ["revenue", "operating", "liquidity", "results", "cash",
                         "quarter", "year", "income", "loss", "expense", "financial",
                         "increased", "decreased", "compared", "net"],
        }

    return {"risk_factors": risk_spec, "mda": mda_spec}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_sections(
    html: str,
    accession: str = "",
    filing_date: str = "",
    form_type: str = "10-Q",
) -> ExtractionResult:
    soup            = BeautifulSoup(html, "lxml")
    _strip_boilerplate(soup)
    full_text       = soup.get_text(separator="\n", strip=True)
    full_char_count = len(full_text)
    specs           = _build_specs(form_type)

    risk_result = _extract_section(soup, full_text, "risk_factors", specs, accession)
    mda_result  = _extract_section(soup, full_text, "mda",          specs, accession)
    gaps        = _identify_gaps(risk_result, mda_result, full_char_count)

    return ExtractionResult(
        filing_accession=accession,
        filing_date=filing_date,
        form_type=form_type,
        risk_factors=risk_result,
        mda=mda_result,
        full_doc_char_count=full_char_count,
        known_gaps=gaps,
    )


# ---------------------------------------------------------------------------
# Per-section extraction — tries strategies in order
# ---------------------------------------------------------------------------

def _extract_section(
    soup: BeautifulSoup, full_text: str,
    section_key: str, specs: dict, accession: str,
) -> SectionResult:
    spec = specs[section_key]

    text = _try_anchor_strategy(soup, spec)
    if text and _plausible(text, spec):
        return _ok(section_key, spec, text, ExtractionMethod.ANCHOR_HREF, 0.92)

    text = _try_heading_strategy(soup, spec)
    if text and _plausible(text, spec):
        return _ok(section_key, spec, text, ExtractionMethod.HEADING_TAG, 0.85)

    text = _try_ixbrl_div_strategy(soup, spec)
    if text and _plausible(text, spec):
        return _ok(section_key, spec, text, ExtractionMethod.IXBRL_DIV, 0.82)

    text = _try_toc_strategy(soup, spec)
    if text and _plausible(text, spec):
        return _ok(section_key, spec, text, ExtractionMethod.TABLE_OF_CONTENTS, 0.78)

    text = _try_pattern_strategy(full_text, spec)
    if text and _plausible(text, spec):
        return _ok(section_key, spec, text, ExtractionMethod.PATTERN_MATCH, 0.70)

    gap = (
        f"{spec['display']} ({spec['item_label']}) could not be isolated "
        f"from accession {accession}. Returning full document text as fallback."
    )
    return SectionResult(
        section_name=section_key,
        item_label=spec["item_label"],
        text=full_text,
        method=ExtractionMethod.RAW_FALLBACK,
        extraction_success=False,
        failure_reason=f"All extraction strategies failed for {spec['display']}",
        confidence_score=0.20,
        coverage_gap_note=gap,
    )


# ---------------------------------------------------------------------------
# Strategy 1: anchor / id
# ---------------------------------------------------------------------------

def _try_anchor_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    item_norm = spec["item_label"].lower().replace(" ", "")
    for tag in soup.find_all(True, attrs=True):
        for attr in ("name", "id"):
            val = re.sub(r"[\s\-_]", "", (tag.get(attr) or "").lower())
            if item_norm in val:
                text = _collect_forward(tag, spec["next_items"])
                if text and len(text) >= MIN_SECTION_CHARS:
                    return text
                break
    return None


# ---------------------------------------------------------------------------
# Strategy 2: semantic heading tags (h1-h4, b, strong)
# ---------------------------------------------------------------------------

def _try_heading_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    for selector in ["h1", "h2", "h3", "h4", "b", "strong"]:
        for tag in soup.select(selector):
            if start_re.search(_norm(tag.get_text())) and not _looks_like_toc_entry(tag):
                text = _collect_forward(tag, spec["next_items"])
                # Skip if too short — likely a TOC hit, not the real section
                if text and len(text) >= MIN_SECTION_CHARS:
                    return text
    return None


# ---------------------------------------------------------------------------
# Strategy 3: iXBRL bold div/span
# ---------------------------------------------------------------------------

_BOLD_STYLE_RE = re.compile(r"font-weight\s*:\s*(bold|700|800|900)", re.IGNORECASE)


def _try_ixbrl_div_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    """
    Modern EDGAR iXBRL filings use <div style="font-weight:bold"> for headings.

    KEY FIX: EDGAR TOC contains bold Item headings followed immediately by the
    next Item heading. We skip any match whose collected text is too short
    (< MIN_SECTION_CHARS), which means we hit a TOC entry, and keep scanning
    for the real section heading further down the document.
    """
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    best_text = None

    for tag in soup.find_all(["div", "span", "p", "td"]):
        if not _BOLD_STYLE_RE.search(tag.get("style", "")):
            continue
        if not start_re.search(_norm(tag.get_text())):
            continue
        if _looks_like_toc_entry(tag):
            continue

        text = _collect_forward(tag, spec["next_items"])
        if not text:
            continue

        if len(text) >= MIN_SECTION_CHARS:
            # Good hit — real section content
            return text

        # Text too short — this was a TOC entry or stub heading.
        # Keep the best short result as a fallback but keep scanning.
        if best_text is None or len(text) > len(best_text):
            best_text = text

    # Return best short result only if nothing better was found
    # (will fail _plausible check and fall through to next strategy)
    return best_text


# ---------------------------------------------------------------------------
# Strategy 4: TOC link traversal
# ---------------------------------------------------------------------------

def _try_toc_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    start_re  = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    target_id = None
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("#") and start_re.search(_norm(a.get_text())):
            target_id = a["href"][1:]
            break
    if not target_id:
        return None
    target = soup.find(id=target_id) or soup.find(attrs={"name": target_id})
    if not target:
        return None
    text = _collect_forward(target, spec["next_items"])
    return text if text and len(text) >= MIN_SECTION_CHARS else None


# ---------------------------------------------------------------------------
# Strategy 5: plain text pattern match
# ---------------------------------------------------------------------------

def _try_pattern_strategy(full_text: str, spec: dict) -> Optional[str]:
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    end_re   = re.compile("|".join(spec["end_patterns"]),   re.IGNORECASE)

    # Find ALL occurrences of the start pattern and take the one with the most content
    best_text = None
    search_from = 0
    while True:
        m = start_re.search(full_text, search_from)
        if not m:
            break
        end_m = end_re.search(full_text, m.end())
        if end_m:
            candidate = full_text[m.start():end_m.start()].strip()
        else:
            candidate = full_text[m.start(): m.start() + 80_000].strip()

        if len(candidate) >= MIN_SECTION_CHARS:
            if best_text is None or len(candidate) > len(best_text):
                best_text = candidate
            # Take the first substantial hit
            break

        search_from = m.end()

    return best_text


# ---------------------------------------------------------------------------
# Shared forward text collector
# ---------------------------------------------------------------------------

def _collect_forward(
    start_tag,
    end_items: list[str],
    max_chars: int = 150_000,
) -> Optional[str]:
    """
    Walk forward in the document from start_tag collecting text.
    Stops when a STANDALONE section heading for a next-item is found.

    Cross-reference guard: "See Item 3 below" inside running prose does NOT
    stop collection — only standalone bold headings do.
    """
    try:
        all_tags = list(start_tag.find_all_next(True))
    except Exception:
        return None

    chunks = []
    total  = 0

    for tag in all_tags:
        if total >= max_chars:
            break

        tag_text = tag.get_text(separator=" ", strip=True) \
            if hasattr(tag, "get_text") else str(tag)
        tag_norm = _norm(tag_text)

        # Only stop on STANDALONE heading elements — not inline text
        if _looks_like_standalone_heading(tag):
            for end_item in end_items:
                if tag_norm.startswith(end_item) or f"\n{end_item}" in tag_norm:
                    result = "\n".join(chunks).strip()
                    return result if result else None

        snippet = tag_text.strip()
        if snippet:
            chunks.append(snippet)
            total += len(snippet)

    result = "\n".join(chunks).strip()
    return result if result else None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "ix:header", "ix:hidden"]):
        tag.decompose()


def _plausible(text: str, spec: dict) -> bool:
    """Check that extracted text plausibly belongs to the target section."""
    if not text or len(text) < 200:
        return False
    text_lower = text.lower()
    hits = sum(1 for kw in spec["keywords"] if kw in text_lower)
    return hits >= 2


def _looks_like_toc_entry(tag) -> bool:
    """True if tag looks like a TOC line — short text ending with a page number."""
    text = tag.get_text(strip=True)
    return len(text) < 80 and bool(re.search(r"\d{1,3}\s*$", text))


def _looks_like_standalone_heading(tag) -> bool:
    """
    True only for structural heading elements.
    Inline <b> or <span> inside paragraphs are NOT standalone headings —
    this prevents cross-references like "see Item 3 below" from stopping collection.
    """
    if not hasattr(tag, "name"):
        return False
    # Unambiguous heading tags
    if tag.name in ("h1", "h2", "h3", "h4"):
        return True
    # Bold div/span — only standalone if it's a direct child of body/section,
    # not nested inside a paragraph. Check: parent is not a <p> or <td>.
    if tag.name in ("div", "span"):
        if _BOLD_STYLE_RE.search(tag.get("style", "")):
            text = tag.get_text(strip=True)
            parent = tag.parent
            parent_name = getattr(parent, "name", "") if parent else ""
            # Short bold div not inside a paragraph → standalone heading
            if len(text) < 150 and parent_name not in ("p", "li", "span"):
                return True
    # Bold tags — only standalone if they ARE the paragraph (not inline in prose)
    if tag.name in ("b", "strong"):
        text = tag.get_text(strip=True)
        parent = tag.parent
        parent_name = getattr(parent, "name", "") if parent else ""
        # If the bold is the entire paragraph, it's a heading
        if parent_name == "p" and len(parent.get_text(strip=True)) == len(text):
            return True
    return False


def _ok(
    section_key: str, spec: dict, text: str,
    method: ExtractionMethod, confidence: float,
) -> SectionResult:
    return SectionResult(
        section_name=section_key, item_label=spec["item_label"],
        text=text, method=method, extraction_success=True,
        confidence_score=confidence,
    )


def _identify_gaps(
    risk: SectionResult, mda: SectionResult, full_char_count: int,
) -> list[str]:
    gaps = []
    if not risk.extraction_success:
        gaps.append(f"Risk Factors could not be isolated (method: {risk.method.value})")
    if not mda.extraction_success:
        gaps.append(f"MD&A could not be isolated (method: {mda.method.value})")
    if full_char_count < 5_000:
        gaps.append("Filing HTML is unusually short — may be a stub or redirect")
    if full_char_count > 2_000_000:
        gaps.append("Filing HTML is very large (>2MB) — extraction may have boundary errors")
    return gaps