"""
scorer.py — RiskLens materiality scorer
Scans extracted sections and delta changes for high-risk financial/legal signals.
All outputs are explicitly labeled as ESTIMATES.

SCORING CALIBRATION FIX:
- Absolute keyword presence is weighted very low (large filings always have risk words)
- New signals (appeared in newer filing but not older) get 3x weight
- Signals found in changed sentences get 2x weight
- Raw score thresholds raised so "critical" is rare and meaningful
- A filing with no material changes should score LOW even if it has many risk keywords
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from delta import SectionDelta, ChangeMagnitude


class MaterialityLevel(str, Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Signal tiers
# ---------------------------------------------------------------------------

TIER1_SIGNALS = [
    "going concern", "material weakness", "restatement", "default",
    "covenant violation", "insolvency", "bankruptcy", "SEC investigation",
    "DOJ", "criminal", "class action", "fraud", "cybersecurity incident",
    "data breach", "force majeure", "impairment", "goodwill impairment",
    "write-off", "write-down", "regulatory action",
]

TIER2_SIGNALS = [
    "significant uncertainty", "substantial doubt", "adverse",
    "contingent liability", "unfavorable", "declining revenue",
    "revenue decline", "net loss", "operating loss", "cash burn",
    "negative cash flow", "restructuring", "layoff", "workforce reduction",
    "supply chain disruption", "tariff", "trade restriction", "sanctions",
    "litigation", "settlement", "injunction", "compliance failure",
    "impairment charge", "liquidity risk", "interest rate risk",
    "foreign exchange", "currency risk", "inflation",
]

TIER3_SIGNALS = [
    "may", "could", "risk", "uncertain", "volatility", "competition",
    "macroeconomic", "recession", "slowdown", "customer concentration",
    "key personnel", "cybersecurity", "technology risk", "execution risk",
    "integration risk", "acquisition", "leverage", "refinancing", "dilution",
]

DISCLAIMER = (
    "ESTIMATE — All materiality assessments, signal detections, and analyst notes "
    "produced by RiskLens are automated estimates based on keyword matching and "
    "statistical heuristics. They do not constitute financial, legal, or investment advice. "
    "Independent verification by a qualified analyst is required before any reliance on these outputs."
)


@dataclass
class SignalHit:
    signal: str
    tier: int
    weight: int
    context: str
    in_change: bool


@dataclass
class SectionScore:
    section_name: str
    materiality: MaterialityLevel
    raw_score: float
    tier1_hits: list[SignalHit]
    tier2_hits: list[SignalHit]
    tier3_hits: list[SignalHit]
    new_signals: list[SignalHit]
    removed_signals: list[SignalHit]
    analyst_note: str
    is_estimate: bool = True  # Always True — never remove


@dataclass
class ScoringResult:
    risk_factors: SectionScore
    mda: SectionScore
    overall_materiality: MaterialityLevel
    top_signals: list[str]
    disclaimer: str = field(default_factory=lambda: DISCLAIMER)
    scoring_success: bool = True
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_sections(
    newer_risk_text: Optional[str],
    older_risk_text: Optional[str],
    newer_mda_text: Optional[str],
    older_mda_text: Optional[str],
    risk_delta: Optional[SectionDelta] = None,
    mda_delta: Optional[SectionDelta] = None,
) -> ScoringResult:
    try:
        risk_score = _score_section(
            "risk_factors", newer_risk_text, older_risk_text, risk_delta
        )
        mda_score = _score_section(
            "mda", newer_mda_text, older_mda_text, mda_delta
        )
        overall  = _combined_materiality(risk_score.materiality, mda_score.materiality)
        top      = _top_signals(risk_score, mda_score)
        return ScoringResult(
            risk_factors=risk_score, mda=mda_score,
            overall_materiality=overall, top_signals=top,
        )
    except Exception as exc:
        empty = _empty_score("unknown")
        return ScoringResult(
            risk_factors=empty, mda=empty,
            overall_materiality=MaterialityLevel.LOW,
            top_signals=[], scoring_success=False,
            failure_reason=f"Scoring error: {exc}",
        )


# ---------------------------------------------------------------------------
# Section scoring
# ---------------------------------------------------------------------------

def _score_section(
    section_name: str,
    newer_text: Optional[str],
    older_text: Optional[str],
    delta: Optional[SectionDelta],
) -> SectionScore:
    if not newer_text:
        return _empty_score(section_name)

    # All signals in newer text (base weight: low, these are always present)
    newer_hits = _find_signals(newer_text, in_change=False)

    # Signals in changed sentences only (higher weight — these are new language)
    if delta and delta.delta_success:
        changed_text = _extract_changed_text(delta)
        if changed_text:
            change_hits = {h.signal for h in _find_signals(changed_text, in_change=True)}
            for h in newer_hits:
                if h.signal in change_hits:
                    h.in_change = True

    # New signals: present in newer but not older
    new_signals: list[SignalHit] = []
    removed_signals: list[SignalHit] = []
    if older_text:
        older_signal_names = {h.signal for h in _find_signals(older_text, in_change=False)}
        newer_signal_names = {h.signal for h in newer_hits}
        new_names     = newer_signal_names - older_signal_names
        removed_names = older_signal_names - newer_signal_names
        new_signals     = [h for h in newer_hits if h.signal in new_names]
        removed_signals = [
            h for h in _find_signals(older_text, in_change=False)
            if h.signal in removed_names
        ]

    raw        = _compute_raw_score(newer_hits, new_signals, delta)
    materiality = _score_to_materiality(raw)

    t1 = [h for h in newer_hits if h.tier == 1]
    t2 = [h for h in newer_hits if h.tier == 2]
    t3 = [h for h in newer_hits if h.tier == 3]
    note = _analyst_note(section_name, materiality, t1, new_signals, removed_signals, delta)

    return SectionScore(
        section_name=section_name,
        materiality=materiality,
        raw_score=round(raw, 2),
        tier1_hits=t1[:20], tier2_hits=t2[:20], tier3_hits=t3[:30],
        new_signals=new_signals[:15],
        removed_signals=removed_signals[:15],
        analyst_note=note,
        is_estimate=True,
    )


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def _find_signals(text: str, in_change: bool) -> list[SignalHit]:
    if not text:
        return []
    text_lower = text.lower()
    hits = []
    seen = set()
    for signal, tier, weight in (
        [(s, 1, 3) for s in TIER1_SIGNALS] +
        [(s, 2, 2) for s in TIER2_SIGNALS] +
        [(s, 3, 1) for s in TIER3_SIGNALS]
    ):
        if signal.lower() in text_lower and signal not in seen:
            seen.add(signal)
            hits.append(SignalHit(
                signal=signal, tier=tier, weight=weight,
                context=_extract_context(text, signal),
                in_change=in_change,
            ))
    return hits


def _extract_context(text: str, signal: str, window: int = 120) -> str:
    idx = text.lower().find(signal.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window // 2)
    end   = min(len(text), idx + len(signal) + window // 2)
    return f"...{text[start:end].replace(chr(10), ' ').strip()}..."


def _extract_changed_text(delta: SectionDelta) -> str:
    parts = []
    for c in delta.changes:
        if c.change_type in ("added", "rewritten") and c.newer_text:
            parts.append(c.newer_text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Calibrated scoring
#
# Philosophy:
#   - A filing with zero changes and many risk keywords → LOW
#     (every large-cap filing has "may", "could", "risk" hundreds of times)
#   - A filing with new tier-1 signals in changed sentences → HIGH or CRITICAL
#   - Absolute keyword presence contributes minimally to the score
#   - Change magnitude and new signals drive the score up
#
# Thresholds (tuned against 60-ticker live test):
#   LOW      < 8
#   MODERATE 8–19
#   HIGH     20–34
#   CRITICAL >= 35
# ---------------------------------------------------------------------------

def _compute_raw_score(
    hits: list[SignalHit],
    new_signals: list[SignalHit],
    delta: Optional[SectionDelta],
) -> float:
    # Base: absolute keyword presence — low weight to avoid inflation
    # Tier1=1.0, Tier2=0.5, Tier3=0.1 per hit (much lower than before)
    base_weights = {1: 1.0, 2: 0.5, 3: 0.1}
    base = sum(base_weights.get(h.tier, 0) for h in hits)

    # Changed-sentence bonus: signals in new/rewritten text score extra
    change_bonus = sum(
        base_weights.get(h.tier, 0) * 1.5
        for h in hits if h.in_change
    )

    # New signal bonus: signals not present in prior filing score heavily
    # This is where real materiality comes from
    new_weights = {1: 12.0, 2: 6.0, 3: 1.0}
    new_bonus = sum(new_weights.get(h.tier, 0) for h in new_signals)

    # Change magnitude bonus
    mag_bonus = 0.0
    if delta and delta.delta_success:
        mag_bonus = {
            ChangeMagnitude.MINOR:    2.0,
            ChangeMagnitude.MODERATE: 5.0,
            ChangeMagnitude.MAJOR:    10.0,
        }.get(delta.magnitude, 0.0)

    return base + change_bonus + new_bonus + mag_bonus


def _score_to_materiality(score: float) -> MaterialityLevel:
    if score >= 35:
        return MaterialityLevel.CRITICAL
    elif score >= 20:
        return MaterialityLevel.HIGH
    elif score >= 8:
        return MaterialityLevel.MODERATE
    else:
        return MaterialityLevel.LOW


def _combined_materiality(a: MaterialityLevel, b: MaterialityLevel) -> MaterialityLevel:
    order = [MaterialityLevel.LOW, MaterialityLevel.MODERATE,
             MaterialityLevel.HIGH, MaterialityLevel.CRITICAL]
    return max(a, b, key=lambda x: order.index(x))


def _top_signals(risk: SectionScore, mda: SectionScore) -> list[str]:
    all_hits = (
        [(h, "risk_factors") for h in risk.new_signals[:5]]
        + [(h, "mda")         for h in mda.new_signals[:5]]
        + [(h, "risk_factors") for h in risk.tier1_hits[:3]]
        + [(h, "mda")         for h in mda.tier1_hits[:3]]
    )
    seen   = set()
    result = []
    for h, sec in sorted(all_hits, key=lambda x: (-x[0].tier, -x[0].weight)):
        if h.signal not in seen:
            seen.add(h.signal)
            tag = " [NEW]" if h in [] else ""
            result.append(f"{h.signal} [{sec}]")
    return result[:10]


# ---------------------------------------------------------------------------
# Analyst note
# ---------------------------------------------------------------------------

def _analyst_note(
    section_name: str, materiality: MaterialityLevel,
    tier1_hits: list[SignalHit], new_signals: list[SignalHit],
    removed_signals: list[SignalHit], delta: Optional[SectionDelta],
) -> str:
    label = "Risk Factors" if section_name == "risk_factors" else "MD&A"
    parts = [f"[ESTIMATE] {label} materiality: {materiality.value.upper()}."]

    if tier1_hits:
        names = ", ".join(h.signal for h in tier1_hits[:5])
        parts.append(f"Critical signals present: {names}.")

    if new_signals:
        names = ", ".join(h.signal for h in new_signals[:5])
        parts.append(f"NEW vs prior filing: {names}.")

    if removed_signals:
        names = ", ".join(h.signal for h in removed_signals[:5])
        parts.append(f"Removed vs prior filing: {names}.")

    if delta and delta.delta_success:
        parts.append(
            f"Section changed {delta.pct_changed * 100:.1f}% by sentence "
            f"(magnitude: {delta.magnitude.value})."
        )

    parts.append(
        "Automated estimate only — do not use as sole basis for any investment or legal decision."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_score(section_name: str) -> SectionScore:
    return SectionScore(
        section_name=section_name,
        materiality=MaterialityLevel.LOW, raw_score=0.0,
        tier1_hits=[], tier2_hits=[], tier3_hits=[],
        new_signals=[], removed_signals=[],
        analyst_note=(
            f"[ESTIMATE] No text available for {section_name} — scoring skipped. "
            "Check extraction logs for failure details."
        ),
        is_estimate=True,
    )