from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from app import main
from app.config import Settings
from app.requirements import (
    extract_unlisted_requirements,
    filter_known_skill_requirements,
)
from app.schemas import AnalyzeResult, ScoreBreakdown
from app.scoring import (
    ScoreInputs,
    apply_display_map,
    combine,
    combine_with_signal,
    combine_v2,
    v2_skill_coverage,
)


def test_apply_display_map() -> None:
    spec = "0:0,0.274424:70,0.449568:85,1:100"

    assert apply_display_map(0.5, "") == 50
    assert apply_display_map(0.274424, spec) == 70
    assert apply_display_map(0.449568, spec) == 85
    assert apply_display_map(
        (0.274424 + 0.449568) / 2, spec
    ) == 78
    assert apply_display_map(-1.0, spec) == 0
    assert apply_display_map(2.0, spec) == 100


@pytest.mark.parametrize(
    "spec",
    (
        "0:0",
        "zero:0,1:100",
        "0:-1,1:100",
        "0:0,1:101",
        "0:0,0.5:40,0.4:50,1:100",
        "0:0,0.5:60,1:50",
        "0:0,nan,1:100",
    ),
)
def test_settings_reject_invalid_display_map(spec: str) -> None:
    with pytest.raises(ValueError):
        Settings(scoring_display_map=spec)


def _inputs(
        *,
        raw: float = 0.75,
        section: float = 0.5,
        skills: float = 0.5,
        hard: float = 4.0,
        hard_coverage: float = 0.5,
        soft: int = 0,
        matched: int = 2,
        requirements: tuple[float, ...] = (),
) -> ScoreInputs:
    return ScoreInputs(
        raw_semantic=raw,
        section_raw=section,
        skills_coverage=skills,
        hard_coverage=hard_coverage,
        hard_weight_sum=hard,
        hard_skill_count=int(hard),
        soft_skill_count=soft,
        hard_matched_count=matched,
        requirement_scores=requirements,
    )


def test_legacy_golden_score_remains_unchanged() -> None:
    parts = combine(0.75, 0.0, 1.0, Settings())

    assert parts.match_percentage == 60
    assert parts.final == pytest.approx(0.6)


def test_v2_golden_scores_by_hard_weight_sum() -> None:
    settings = Settings(scoring_mode="v2")
    scores = [
        combine_v2(
            _inputs(hard=float(hard_weight)),
            settings,
        ).final
        for hard_weight in range(1, 9)
    ]

    assert scores == pytest.approx(
        [0.82, 0.729, 0.643, 0.562, 0.562, 0.562, 0.562, 0.562],
        abs=0.0005,
    )


def test_zero_hard_matches_are_gated_in_v2() -> None:
    settings = Settings(scoring_mode="v2")
    parts = combine_v2(
        _inputs(
            raw=1.0,
            section=1.0,
            skills=0.0,
            hard=3.0,
            hard_coverage=0.0,
            matched=0,
        ),
        settings,
    )

    assert combine_with_signal(
        1.0, 0.0, 1.0, Settings(), 3, 0, 0
    ).match_percentage == 100
    assert parts.final <= 0.55


def test_zero_hard_matches_with_full_gate_stay_below_bound() -> None:
    settings = Settings(scoring_mode="v2")
    for semantic in np.linspace(0.0, 1.0, 5):
        for section in np.linspace(0.0, 1.0, 5):
            for requirement in np.linspace(0.0, 1.0, 5):
                parts = combine_v2(
                    _inputs(
                        raw=semantic,
                        section=section,
                        skills=0.0,
                        hard=4.0,
                        hard_coverage=0.0,
                        matched=0,
                        requirements=(requirement,),
                    ),
                    settings,
                )
                assert parts.final <= 0.40 + 1e-9


def test_reliability_continuity_random_profiles() -> None:
    settings = Settings(scoring_mode="v2")
    rng = np.random.default_rng(0)
    step = 0.25
    bound = (
        settings.skill_weight
        / float(settings.v2_weight_reliability_skills)
        + (1.0 - settings.gate_floor)
        / float(settings.v2_gate_reliability_skills)
    ) * step
    largest_ratio = 0.0
    for _ in range(200):
        raw, section, skills, hard_coverage = rng.random(4)
        requirement_scores = tuple(
            rng.random(int(rng.integers(0, 4)))
        )
        scores = [
            combine_v2(
                _inputs(
                    raw=raw,
                    section=section,
                    skills=skills,
                    hard=hard_weight,
                    hard_coverage=hard_coverage,
                    requirements=requirement_scores,
                ),
                settings,
            ).final
            for hard_weight in np.arange(0.0, 8.0 + step, step)
        ]
        largest_ratio = max(
            largest_ratio,
            max(
                abs(right - left) / bound
                for left, right in zip(scores, scores[1:])
            ),
        )
        assert all(
            abs(right - left) <= bound + 1e-9
            for left, right in zip(scores, scores[1:])
        )
    print(f"largest continuity ratio: {largest_ratio:.6f}")

    profile = dict(raw=0.75, section=0.5, skills=0.0, hard_coverage=0.0)
    for hard_weight in (0, 1, 2, 3, 4):
        score = combine_v2(
            _inputs(**profile, hard=float(hard_weight)),
            settings,
        ).final
        print(f"zero-match profile hard_weight={hard_weight}: {score:.6f}")


def test_v2_monotonic_sweeps_with_and_without_calibrator() -> None:
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit(
        np.asarray([0.1, 0.3, 0.5, 0.8]),
        np.asarray([0.0, 0.2, 0.7, 1.0]),
    )
    for active_r in (0.0, 0.5, 1.0):
        for fitted in (False, True):
            settings = Settings(
                scoring_mode="v2",
                semantic_source="calibrated",
            )
            scale = float(settings.v2_weight_reliability_skills)
            hard_weight = active_r * scale

            def score(**kwargs: object) -> float:
                return combine_v2(
                    _inputs(hard=hard_weight, **kwargs),
                    settings,
                    calibrator if fitted else None,
                ).final

            for field in ("raw", "section", "skills", "hard_coverage"):
                values = [
                    score(**{field: value})
                    for value in np.linspace(0.0, 1.0, 21)
                ]
                assert values == sorted(values)
            values = [
                score(requirements=(value,))
                for value in np.linspace(0.0, 1.0, 21)
            ]
            assert values == sorted(values)


def test_semantic_skill_credit_is_below_lexical_credit() -> None:
    kwargs = {
        "jd_weights": {"python": 1.0},
        "skill_types": {"python": "hard"},
        "evidence_weights": {"python": 1.0},
        "semantic_credit": 0.6,
    }
    lexical = v2_skill_coverage(
        **kwargs, lexical_matches={"python"}, semantic_matches=set()
    )
    semantic = v2_skill_coverage(
        **kwargs, lexical_matches=set(), semantic_matches={"python"}
    )

    assert semantic[0] < lexical[0]
    assert semantic[1] < lexical[1]
    assert semantic[2] == lexical[2] == 1


def test_thin_lexicon_requirements_keep_weight_and_skill_credit() -> None:
    settings = Settings(scoring_mode="v2")
    low = combine_v2(
        _inputs(
            hard=0.0,
            hard_coverage=0.0,
            skills=0.0,
            requirements=(0.1,),
        ),
        settings,
    )
    high = combine_v2(
        _inputs(
            hard=0.0,
            hard_coverage=0.0,
            skills=0.0,
            requirements=(0.9,),
        ),
        settings,
    )

    assert low.r_weight == 0.0
    assert low.requirement_weight > 0.0
    assert high.final > low.final


def test_v2_weights_sum_to_one_and_scores_are_bounded() -> None:
    settings = Settings(scoring_mode="v2")
    for requirements in ((), (0.2, 0.8)):
        parts = combine_v2(
            _inputs(requirements=requirements),
            settings,
        )
        total = (
            parts.semantic_weight
            + parts.skill_weight
            + parts.section_weight
            + parts.requirement_weight
        )
        assert total == pytest.approx(1.0)
        assert 0.0 <= parts.final <= 1.0


def test_v2_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        Settings(scoring_mode="other")
    with pytest.raises(ValueError):
        Settings(gate_floor=1.1)
    with pytest.raises(ValueError):
        Settings(gate_gamma=0.0)
    with pytest.raises(ValueError):
        Settings(semantic_skill_credit=-0.1)
    with pytest.raises(ValueError):
        Settings(unlisted_requirement_share=0.6)
    with pytest.raises(ValueError):
        Settings(v2_phrase_cap=0)
    with pytest.raises(ValueError):
        Settings(section_low=1.0, section_high=1.0)
    with pytest.raises(ValueError):
        Settings(v2_weight_reliability_skills=0.0)
    with pytest.raises(ValueError):
        Settings(v2_gate_reliability_skills=-1.0)


def test_v2_reliability_scales_are_configurable() -> None:
    settings = Settings(
        scoring_mode="v2",
        v2_weight_reliability_skills=2.0,
        v2_gate_reliability_skills=2.0,
    )
    parts = combine_v2(_inputs(hard=2.0), settings)
    assert parts.r_weight == 1.0
    assert parts.r_gate == 1.0


def test_requirement_filter_matches_prepare_score_in_both_modes() -> None:
    from app.skills import load_skills

    lexicon = load_skills()
    jd = (
        "We need experience with python scripting, and knowledge of a "
        "proprietary kiln fixture for production work."
    )
    extracted = extract_unlisted_requirements(jd, lexicon)
    kept, dropped = filter_known_skill_requirements(extracted, lexicon)

    assert "python scripting" in dropped
    assert "python scripting" not in kept

    class StubModel:
        max_seq_length = 256

    resume = "Experience " + ("delivered production systems. " * 20)
    v2_prep = main._prepare_score(
        resume, "", jd, Settings(scoring_mode="v2"), StubModel(), lexicon
    )
    legacy_prep = main._prepare_score(
        resume, "", jd, Settings(scoring_mode="legacy"),
        StubModel(), lexicon,
    )
    assert "python scripting" not in v2_prep.unlisted_requirements
    assert v2_prep.dropped_requirement_count >= 1
    assert "python scripting" in legacy_prep.unlisted_requirements


def test_spread_grid_has_distinct_non_saturated_scores() -> None:
    # Checks output granularity only, not real score spread.
    settings = Settings(scoring_mode="v2")
    scores = [
        combine_v2(
            _inputs(
                raw=raw,
                section=section,
                skills=skills,
                hard=hard,
                hard_coverage=hard_coverage,
                matched=matched,
            ),
            settings,
        ).match_percentage
        for raw in np.linspace(0.2, 0.9, 6)
        for section in np.linspace(0.1, 0.9, 5)
        for skills, hard_coverage, hard, matched in (
            (0.0, 0.0, 4.0, 0),
            (0.4, 0.4, 5.0, 2),
            (0.8, 0.8, 7.0, 5),
        )
    ]
    assert len(set(scores)) >= 40
    assert sum(score >= 95 for score in scores) / len(scores) < 0.10


def test_unrounded_ranking_breaks_rounded_score_ties() -> None:
    def result(
            name: str,
            score: float,
            hard: float,
            raw: float,
    ) -> AnalyzeResult:
        return AnalyzeResult(
            filename=name,
            candidate_name=name,
            match_percentage=round(score * 100),
            top_skills=[],
            missing_skills=[],
            score_breakdown=ScoreBreakdown(
                semantic=0, skills=0, section_fit=0
            ),
            reason="",
            flags=[],
            status="ok",
            ranking_score=score,
            ranking_hard_coverage=hard,
            ranking_raw_semantic=raw,
        )

    candidates = [
        result("lower", 0.804, 0.9, 0.9),
        result("higher", 0.805, 0.1, 0.1),
    ]
    ordered = sorted(
        candidates,
        key=main.ranking_key,
        reverse=True,
    )

    assert candidates[0].match_percentage == candidates[1].match_percentage
    assert [item.filename for item in ordered] == ["higher", "lower"]

    tied_unrounded = [
        result("low-hard", 0.8, 0.1, 0.9),
        result("high-hard", 0.8, 0.9, 0.1),
        result("high-raw", 0.8, 0.9, 0.9),
    ]
    ordered = sorted(tied_unrounded, key=main.ranking_key, reverse=True)
    assert [item.filename for item in ordered] == [
        "high-raw", "high-hard", "low-hard"
    ]
