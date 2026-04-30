"""
test_unit.py — RiskLens unit tests (no network required)
Tests extractor, delta, and scorer against synthetic and real-structure HTML.

Run with: pytest test_unit.py -v
"""

try:
    import pytest
except ImportError:
    pytest = None  # Tests run without pytest via run_tests() below

from extractor import (
    ExtractionMethod, SectionResult, extract_sections,
    _norm, _plausible,
)
from delta import compute_delta, ChangeMagnitude, _sentence_similarity, _to_sentences
from scorer import score_sections, MaterialityLevel, DISCLAIMER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_10Q_HTML = """
<html><body>
<h1>FORM 10-Q</h1>
<p>PART I — FINANCIAL INFORMATION</p>

<h2>Item 1A. Risk Factors</h2>
<p>We face significant competition from larger, better-capitalized companies that may
adversely affect our business and operating results. There is no assurance that we
will be able to maintain our market position. We may be unable to attract and retain
key personnel necessary to execute our strategy. Regulatory changes could materially
harm our revenue and liquidity. We have significant debt obligations that may limit
our operating flexibility. Our customers may reduce spending due to macroeconomic
uncertainty. Cybersecurity incidents could expose us to material liability and reputational harm.
Supply chain disruptions could increase our costs and delay product delivery.</p>

<p>Our operations are subject to foreign exchange risk and interest rate volatility.
Litigation outcomes are inherently uncertain and adverse rulings could materially
impact our financial condition. We depend on a small number of customers for a
substantial portion of our revenue, creating customer concentration risk.</p>

<h2>Item 1B. Unresolved Staff Comments</h2>
<p>None.</p>

<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>Revenue for the quarter was $500 million, an increase of 12% compared to the prior year period.
Operating income was $80 million, compared to $65 million in the prior year period.
Net income was $55 million. Cash and cash equivalents at quarter end were $320 million.
Liquidity remained strong with access to our $500 million revolving credit facility.
We generated $95 million in operating cash flow during the quarter.
Capital expenditures were $22 million. We expect continued growth in our core markets.</p>

<p>Results of operations reflect increased demand across our product lines. 
Cost of revenue increased 9% year-over-year due to supply chain pressures.
Research and development expense was $45 million for the quarter.
Selling, general and administrative expenses were $130 million.</p>

<h2>Item 3. Quantitative and Qualitative Disclosures About Market Risk</h2>
<p>See attached financial statements.</p>
</body></html>
"""

MINIMAL_10Q_HTML_V2 = """
<html><body>
<h1>FORM 10-Q</h1>

<h2>Item 1A. Risk Factors</h2>
<p>We face significant competition from larger, better-capitalized companies that may
adversely affect our business and operating results. There is no assurance that we
will be able to maintain our market position. We may be unable to attract and retain
key personnel necessary to execute our strategy. Regulatory changes could materially
harm our revenue and liquidity. We have significant debt obligations that may limit
our operating flexibility. Cybersecurity incidents could expose us to material liability.</p>

<p>NEW RISK: We are subject to a going concern qualification from our auditors due to
recurring net losses and negative cash flow from operations. Our material weakness
in internal controls over financial reporting may result in a restatement of prior
period financial statements. Class action litigation has been filed against the company
alleging securities fraud.</p>

<p>Our operations are subject to foreign exchange risk and interest rate volatility.
We depend on a small number of customers for a substantial portion of our revenue.</p>

<h2>Item 1B. Unresolved Staff Comments</h2>
<p>None.</p>

<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>Revenue for the quarter was $420 million, a decline of 16% compared to the prior year period.
Operating loss was $(25) million, compared to operating income of $80 million in the prior year.
Net loss was $(48) million. Cash and cash equivalents at quarter end were $85 million.
Liquidity has deteriorated significantly. We have drawn down $450 million of our credit facility.
We generated negative cash flow from operations of $(35) million during the quarter.
Capital expenditures were reduced to $8 million as part of cost-reduction initiatives.
We are evaluating strategic alternatives including potential asset sales and restructuring.</p>

<h2>Item 3. Quantitative Disclosures About Market Risk</h2>
<p>See below.</p>
</body></html>
"""

STUB_HTML = "<html><body><p>Loading...</p></body></html>"

HTML_NO_SECTIONS = """
<html><body>
<h1>Annual Report</h1>
<p>This is a filing without standard item numbering. The company faces various business risks
including competition, regulatory changes, and macroeconomic uncertainty. Revenue was strong
this year. We maintain adequate liquidity. Operating results were satisfactory.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------

class TestExtractor:

    def test_extracts_risk_factors_from_clean_html(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        rf = result.risk_factors
        assert rf.extraction_success is True
        assert rf.text is not None
        assert len(rf.text) > 200
        assert "risk" in rf.text.lower() or "competition" in rf.text.lower()

    def test_extracts_mda_from_clean_html(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        mda = result.mda
        assert mda.extraction_success is True
        assert mda.text is not None
        assert len(mda.text) > 200
        assert "revenue" in mda.text.lower() or "operating" in mda.text.lower()

    def test_risk_factors_does_not_bleed_into_mda(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        rf = result.risk_factors
        if rf.extraction_success:
            # MD&A-specific text should not appear in Risk Factors
            assert "revenue for the quarter" not in rf.text.lower()

    def test_mda_does_not_bleed_into_financials(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        mda = result.mda
        if mda.extraction_success:
            assert "see attached financial statements" not in mda.text.lower()

    def test_stub_html_returns_raw_fallback(self):
        result = extract_sections(STUB_HTML, "ACC_STUB", "2024-01-01")
        assert result.risk_factors.extraction_success is False
        assert result.mda.extraction_success is False
        assert result.risk_factors.method == ExtractionMethod.RAW_FALLBACK
        assert result.risk_factors.text is not None  # Raw fallback, not None

    def test_no_sections_returns_raw_fallback_with_reason(self):
        result = extract_sections(HTML_NO_SECTIONS, "ACC002", "2024-01-01")
        # Both should fail clean extraction but still return text
        for section in (result.risk_factors, result.mda):
            assert section.text is not None
            assert section.failure_reason is not None

    def test_empty_html_does_not_crash(self):
        result = extract_sections("", "ACC_EMPTY", "2024-01-01")
        assert result is not None
        assert result.risk_factors.text is not None or result.risk_factors.failure_reason is not None

    def test_per_section_method_tracking(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        assert result.risk_factors.method in ExtractionMethod.__members__.values()
        assert result.mda.method in ExtractionMethod.__members__.values()
        # Methods may differ between sections
        # No assertion they're equal — that's the point

    def test_confidence_score_range(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        for section in (result.risk_factors, result.mda):
            assert 0.0 <= section.confidence_score <= 1.0

    def test_known_gaps_list_present(self):
        result = extract_sections(STUB_HTML, "ACC_STUB", "2024-01-01")
        assert isinstance(result.known_gaps, list)

    def test_char_count_populated(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC001", "2024-09-30")
        assert result.risk_factors.char_count > 0 or not result.risk_factors.extraction_success
        assert result.full_doc_char_count > 0

    def test_provenance_fields_populated(self):
        result = extract_sections(MINIMAL_10Q_HTML, "ACC-PROV", "2024-06-30")
        assert result.filing_accession == "ACC-PROV"
        assert result.filing_date == "2024-06-30"


# ---------------------------------------------------------------------------
# Delta tests
# ---------------------------------------------------------------------------

class TestDelta:

    def setup_method(self):
        r1 = extract_sections(MINIMAL_10Q_HTML, "OLD", "2024-06-30")
        r2 = extract_sections(MINIMAL_10Q_HTML_V2, "NEW", "2024-09-30")
        self.older_risk = r1.risk_factors.text
        self.newer_risk = r2.risk_factors.text
        self.older_mda = r1.mda.text
        self.newer_mda = r2.mda.text

    def test_detects_changes_between_versions(self):
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        assert delta.risk_factors.delta_success is True
        assert delta.risk_factors.added_count + delta.risk_factors.removed_count > 0

    def test_major_change_magnitude_detected(self):
        # V2 adds going concern, class action, material weakness — should be MAJOR or MODERATE
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        assert delta.risk_factors.magnitude in (ChangeMagnitude.MODERATE, ChangeMagnitude.MAJOR)

    def test_mda_significant_change_detected(self):
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        if delta.mda.delta_success:
            assert delta.mda.magnitude != ChangeMagnitude.NONE

    def test_identical_texts_yield_no_changes(self):
        text = self.older_risk
        delta = compute_delta(text, text, self.older_mda, self.older_mda)
        assert delta.risk_factors.magnitude == ChangeMagnitude.NONE
        assert delta.risk_factors.added_count == 0
        assert delta.risk_factors.removed_count == 0

    def test_none_older_returns_failed_delta(self):
        delta = compute_delta(None, self.newer_risk, None, self.newer_mda)
        assert delta.risk_factors.delta_success is False
        assert delta.risk_factors.failure_reason is not None

    def test_none_newer_returns_failed_delta(self):
        delta = compute_delta(self.older_risk, None, self.older_mda, None)
        assert delta.risk_factors.delta_success is False

    def test_comparison_note_present(self):
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        assert "ESTIMATE" in delta.comparison_note

    def test_pct_changed_in_range(self):
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        assert 0.0 <= delta.risk_factors.pct_changed <= 1.0

    def test_sentence_similarity_identical(self):
        assert _sentence_similarity("foo bar baz", "foo bar baz") == 1.0

    def test_sentence_similarity_disjoint(self):
        assert _sentence_similarity("aaa bbb ccc", "xxx yyy zzz") == 0.0

    def test_sentence_similarity_partial(self):
        sim = _sentence_similarity("the company faces significant risk", "the company reported significant revenue")
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        r1 = extract_sections(MINIMAL_10Q_HTML, "OLD", "2024-06-30")
        r2 = extract_sections(MINIMAL_10Q_HTML_V2, "NEW", "2024-09-30")
        self.older_risk = r1.risk_factors.text
        self.newer_risk = r2.risk_factors.text
        self.older_mda = r1.mda.text
        self.newer_mda = r2.mda.text

        from delta import compute_delta
        delta = compute_delta(self.older_risk, self.newer_risk, self.older_mda, self.newer_mda)
        self.risk_delta = delta.risk_factors
        self.mda_delta = delta.mda

    def test_critical_signals_detected_in_v2(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
            self.risk_delta, self.mda_delta,
        )
        # V2 has going concern, material weakness, class action, fraud — should score HIGH or CRITICAL
        assert scoring.risk_factors.materiality in (
            MaterialityLevel.HIGH, MaterialityLevel.CRITICAL
        )

    def test_disclaimer_always_present(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        assert scoring.disclaimer == DISCLAIMER
        assert "ESTIMATE" in scoring.disclaimer

    def test_is_estimate_always_true(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        assert scoring.risk_factors.is_estimate is True
        assert scoring.mda.is_estimate is True

    def test_analyst_note_contains_estimate_label(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        assert "[ESTIMATE]" in scoring.risk_factors.analyst_note
        assert "[ESTIMATE]" in scoring.mda.analyst_note

    def test_new_signals_detected(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
            self.risk_delta, self.mda_delta,
        )
        # V2 has going concern, material weakness not in V1
        new_sig_names = {s.signal for s in scoring.risk_factors.new_signals}
        assert "going concern" in new_sig_names or "material weakness" in new_sig_names

    def test_none_text_returns_low_materiality(self):
        scoring = score_sections(None, None, None, None)
        assert scoring.risk_factors.materiality == MaterialityLevel.LOW
        assert scoring.mda.materiality == MaterialityLevel.LOW
        assert scoring.scoring_success is True

    def test_top_signals_list_populated(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        assert isinstance(scoring.top_signals, list)

    def test_raw_score_non_negative(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        assert scoring.risk_factors.raw_score >= 0
        assert scoring.mda.raw_score >= 0

    def test_overall_materiality_is_max_of_sections(self):
        scoring = score_sections(
            self.newer_risk, self.older_risk,
            self.newer_mda, self.older_mda,
        )
        order = [MaterialityLevel.LOW, MaterialityLevel.MODERATE,
                 MaterialityLevel.HIGH, MaterialityLevel.CRITICAL]
        assert order.index(scoring.overall_materiality) >= max(
            order.index(scoring.risk_factors.materiality),
            order.index(scoring.mda.materiality),
        )


# ---------------------------------------------------------------------------
# Regression: no silent failures
# ---------------------------------------------------------------------------

class TestNoSilentFailures:

    def test_extractor_always_returns_text_or_reason(self):
        for html in ["", STUB_HTML, HTML_NO_SECTIONS, MINIMAL_10Q_HTML]:
            result = extract_sections(html, "TEST", "2024-01-01")
            for section in (result.risk_factors, result.mda):
                # Must have text OR failure_reason — never both None
                assert section.text is not None or section.failure_reason is not None, \
                    f"Section {section.section_name} has neither text nor failure_reason"

    def test_delta_always_returns_structured_output(self):
        for older, newer in [
            (None, None), ("", ""), ("text", None), (None, "text"), ("a b c.", "d e f."),
        ]:
            result = compute_delta(older, newer, older, newer)
            assert result.risk_factors is not None
            assert result.mda is not None

    def test_scorer_never_crashes(self):
        for args in [
            (None, None, None, None),
            ("", "", "", ""),
            ("risk text here", None, "mda text here", None),
        ]:
            result = score_sections(*args)
            assert result is not None
            assert result.disclaimer == DISCLAIMER