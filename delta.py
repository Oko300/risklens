"""
delta.py — RiskLens change analysis
Compares two versions of a section (Risk Factors or MD&A) at sentence level.
Identifies additions, deletions, and rewrites. Classifies change magnitude.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional


class ChangeMagnitude(str, Enum):
    NONE = "none"           # No meaningful differences
    MINOR = "minor"         # Small wording tweaks, < 10% sentences changed
    MODERATE = "moderate"   # 10–30% sentences changed or new material added
    MAJOR = "major"         # > 30% sentences changed, or significant new risk language


@dataclass
class SentenceChange:
    change_type: str        # "added", "removed", "rewritten", "unchanged"
    older_text: Optional[str]
    newer_text: Optional[str]
    similarity: float = 1.0  # For rewrites: 0.0 = completely different


@dataclass
class SectionDelta:
    section_name: str
    magnitude: ChangeMagnitude
    total_older_sentences: int
    total_newer_sentences: int
    added_count: int
    removed_count: int
    rewritten_count: int
    unchanged_count: int
    changes: list[SentenceChange]
    pct_changed: float      # (added + removed + rewritten) / max(older, newer) sentences
    delta_success: bool = True
    failure_reason: Optional[str] = None


@dataclass
class DeltaResult:
    risk_factors: SectionDelta
    mda: SectionDelta
    comparison_note: str = (
        "ESTIMATE: Change analysis is based on automated sentence-level diff. "
        "It may miss semantic equivalences or flag stylistic rewrites as material changes. "
        "Independent verification is recommended before acting on these outputs."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_delta(
    older_risk: Optional[str],
    newer_risk: Optional[str],
    older_mda: Optional[str],
    newer_mda: Optional[str],
) -> DeltaResult:
    """
    Compare old vs new sections. All args may be None if extraction failed;
    delta handles gracefully and sets delta_success=False where needed.
    """
    risk_delta = _compare_section("risk_factors", older_risk, newer_risk)
    mda_delta = _compare_section("mda", older_mda, newer_mda)
    return DeltaResult(risk_factors=risk_delta, mda=mda_delta)


# ---------------------------------------------------------------------------
# Section comparison
# ---------------------------------------------------------------------------

def _compare_section(
    section_name: str,
    older_text: Optional[str],
    newer_text: Optional[str],
) -> SectionDelta:
    if not older_text and not newer_text:
        return _failed_delta(section_name, "Both older and newer text are missing")
    if not older_text:
        return _failed_delta(section_name, "Older filing text missing — cannot compare")
    if not newer_text:
        return _failed_delta(section_name, "Newer filing text missing — cannot compare")

    older_sents = _to_sentences(older_text)
    newer_sents = _to_sentences(newer_text)

    changes = _diff_sentences(older_sents, newer_sents)

    added = sum(1 for c in changes if c.change_type == "added")
    removed = sum(1 for c in changes if c.change_type == "removed")
    rewritten = sum(1 for c in changes if c.change_type == "rewritten")
    unchanged = sum(1 for c in changes if c.change_type == "unchanged")

    max_len = max(len(older_sents), len(newer_sents), 1)
    pct_changed = (added + removed + rewritten) / max_len

    if pct_changed == 0:
        magnitude = ChangeMagnitude.NONE
    elif pct_changed < 0.10:
        magnitude = ChangeMagnitude.MINOR
    elif pct_changed < 0.30:
        magnitude = ChangeMagnitude.MODERATE
    else:
        magnitude = ChangeMagnitude.MAJOR

    return SectionDelta(
        section_name=section_name,
        magnitude=magnitude,
        total_older_sentences=len(older_sents),
        total_newer_sentences=len(newer_sents),
        added_count=added,
        removed_count=removed,
        rewritten_count=rewritten,
        unchanged_count=unchanged,
        changes=changes,
        pct_changed=round(pct_changed, 4),
        delta_success=True,
    )


# ---------------------------------------------------------------------------
# Sentence diffing
# ---------------------------------------------------------------------------

def _diff_sentences(
    older: list[str], newer: list[str]
) -> list[SentenceChange]:
    """
    Use SequenceMatcher to align sentences, then classify each block.
    Rewrites are detected by checking similarity of nearby unmatched sentences.
    """
    matcher = SequenceMatcher(
        isjunk=None, a=older, b=newer, autojunk=False
    )
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                changes.append(SentenceChange(
                    change_type="unchanged",
                    older_text=older[i1 + k],
                    newer_text=newer[j1 + k],
                    similarity=1.0,
                ))
        elif tag == "replace":
            # Try to pair old/new sentences as rewrites vs pure add/remove
            old_block = older[i1:i2]
            new_block = newer[j1:j2]
            changes.extend(_classify_replace_block(old_block, new_block))
        elif tag == "delete":
            for s in older[i1:i2]:
                changes.append(SentenceChange(
                    change_type="removed",
                    older_text=s,
                    newer_text=None,
                    similarity=0.0,
                ))
        elif tag == "insert":
            for s in newer[j1:j2]:
                changes.append(SentenceChange(
                    change_type="added",
                    older_text=None,
                    newer_text=s,
                    similarity=0.0,
                ))
    return changes


def _classify_replace_block(
    old_block: list[str], new_block: list[str]
) -> list[SentenceChange]:
    """
    Given a replace block, try to pair sentences as rewrites if they share
    significant content; otherwise classify as removed/added.
    """
    changes = []
    used_new = set()

    for old_s in old_block:
        best_j = -1
        best_sim = 0.0
        for j, new_s in enumerate(new_block):
            if j in used_new:
                continue
            sim = _sentence_similarity(old_s, new_s)
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_sim >= 0.40:  # Threshold for "rewritten" vs "removed + added"
            used_new.add(best_j)
            changes.append(SentenceChange(
                change_type="rewritten",
                older_text=old_s,
                newer_text=new_block[best_j],
                similarity=round(best_sim, 3),
            ))
        else:
            changes.append(SentenceChange(
                change_type="removed",
                older_text=old_s,
                newer_text=None,
                similarity=0.0,
            ))

    for j, new_s in enumerate(new_block):
        if j not in used_new:
            changes.append(SentenceChange(
                change_type="added",
                older_text=None,
                newer_text=new_s,
                similarity=0.0,
            ))

    return changes


def _sentence_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity — fast and adequate for SEC prose."""
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

# Sentence boundary: end of sentence punctuation followed by whitespace + capital
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')

# Min sentence length to consider (filters out stray punctuation fragments)
_MIN_SENT_LEN = 20


def _to_sentences(text: str) -> list[str]:
    """Split text into sentences. Filters very short fragments."""
    # Normalize whitespace first
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    raw = _SENT_SPLIT.split(text)
    result = []
    for s in raw:
        s = s.strip()
        if len(s) >= _MIN_SENT_LEN:
            result.append(s)
    return result


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, strip punctuation."""
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _failed_delta(section_name: str, reason: str) -> SectionDelta:
    return SectionDelta(
        section_name=section_name,
        magnitude=ChangeMagnitude.NONE,
        total_older_sentences=0,
        total_newer_sentences=0,
        added_count=0,
        removed_count=0,
        rewritten_count=0,
        unchanged_count=0,
        changes=[],
        pct_changed=0.0,
        delta_success=False,
        failure_reason=reason,
    )