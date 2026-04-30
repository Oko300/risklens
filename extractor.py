"""
extractor.py — RiskLens section extractor
Extracts Risk Factors and MD&A from 10-Q / 10-K HTML.
Tracks extraction method and confidence independently per section.
Never fails silently: always returns raw-text fallback + failure flag.

KEY FIX: 10-Q filings label MD&A as "Item 2", not "Item 7".
         Item 7 is the 10-K label only.
         form_type parameter selects the correct item number.
"""

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


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
#   Item 2  MD&A          <--   Item 7   MD&A           <--
#   Item 3  Quant/Qual          Item 7A  Quant/Qual
#   Item 4  Controls            Item 8   Financial Stmts

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
            "start_patterns": [
                r"item\s+2[\.\s\u2014\-\u2013]+management",
                r"item\s+2[\.\s]+discussion",
                r"item\s+2\b",
            ],
            "end_patterns": [
                r"item\s+3\b",
                r"item\s+3[\.\s]+quantitative",
                r"item\s+4\b",
                r"item\s+4[\.\s]+controls",
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
            "start_patterns": [
                r"item\s+7[\.\s\u2014\-\u2013]+management",
                r"item\s+7[\.\s]+discussion",
                r"item\s+7\b",
            ],
            "end_patterns": [
                r"item\s+7a\b",
                r"item\s+8\b",
                r"item\s+8[\.\s]+financial\s+statements",
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
# Strategies
# ---------------------------------------------------------------------------

def _try_anchor_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    item_norm = spec["item_label"].lower().replace(" ", "")
    for tag in soup.find_all(True, attrs=True):
        for attr in ("name", "id"):
            val = re.sub(r"[\s\-_]", "", (tag.get(attr) or "").lower())
            if item_norm in val:
                text = _collect_forward(tag, spec["next_items"])
                if text and len(text) > 100:
                    return text
                break
    return None


def _try_heading_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    for selector in ["h1", "h2", "h3", "h4", "b", "strong"]:
        for tag in soup.select(selector):
            if start_re.search(_norm(tag.get_text())) and not _looks_like_toc_entry(tag):
                text = _collect_forward(tag, spec["next_items"])
                if text:
                    return text
    return None


_BOLD_STYLE_RE = re.compile(r"font-weight\s*:\s*(bold|700|800|900)", re.IGNORECASE)


def _try_ixbrl_div_strategy(soup: BeautifulSoup, spec: dict) -> Optional[str]:
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    for tag in soup.find_all(["div", "span", "p", "td"]):
        if not _BOLD_STYLE_RE.search(tag.get("style", "")):
            continue
        if not start_re.search(_norm(tag.get_text())):
            continue
        if _looks_like_toc_entry(tag):
            continue
        text = _collect_forward(tag, spec["next_items"])
        if text:
            return text
    return None


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
    return _collect_forward(target, spec["next_items"])


def _try_pattern_strategy(full_text: str, spec: dict) -> Optional[str]:
    start_re = re.compile("|".join(spec["start_patterns"]), re.IGNORECASE)
    end_re   = re.compile("|".join(spec["end_patterns"]),   re.IGNORECASE)
    m = start_re.search(full_text)
    if not m:
        return None
    end_m = end_re.search(full_text, m.end())
    if end_m:
        return full_text[m.start():end_m.start()].strip()
    return full_text[m.start(): m.start() + 80_000].strip()


# ---------------------------------------------------------------------------
# Forward collector
# ---------------------------------------------------------------------------

def _collect_forward(
    start_tag, end_items: list[str], max_chars: int = 120_000,
) -> Optional[str]:
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
        if _looks_like_section_heading(tag):
            for end_item in end_items:
                if end_item in tag_norm:
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
    if not text or len(text) < 80:
        return False
    text_lower = text.lower()
    hits = sum(1 for kw in spec["keywords"] if kw in text_lower)
    return hits >= (1 if len(text) < 500 else 2)


def _looks_like_toc_entry(tag) -> bool:
    text = tag.get_text(strip=True)
    return len(text) < 80 and bool(re.search(r"\d{1,3}\s*$", text))


def _looks_like_section_heading(tag) -> bool:
    if not hasattr(tag, "name"):
        return False
    if tag.name in ("h1", "h2", "h3", "h4"):
        return True
    if tag.name in ("b", "strong"):
        return len(tag.get_text(strip=True)) < 150
    if tag.name in ("div", "span", "p", "td"):
        if _BOLD_STYLE_RE.search(tag.get("style", "")):
            return len(tag.get_text(strip=True)) < 150
    if tag.name == "p":
        return bool(tag.find(["b", "strong"]))
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