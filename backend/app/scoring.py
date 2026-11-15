from __future__ import annotations
from dataclasses import dataclass
import joblib
from math import isfinite
import numpy as np
from sklearn.isotonic import IsotonicRegression
from app.config import Settings


@dataclass
class ScoreParts:
    semantic: float
    calibrated_semantic: float
    skill_coverage: float
    section_fit: float
    final: float
    semantic_display: int
    skills_display: int
    section_display: int
    match_percentage: int
    semantic_weight: float = 0.0
    skill_weight: float = 0.0
    section_weight: float = 0.0
    limited_skill_signal: bool = False
    hard_skill_penalty: float = 1.0
    raw_semantic: float = 0.0
    section_raw: float = 0.0
    hard_coverage: float = 0.0
    hard_weight_sum: float = 0.0
    hard_skill_count: int = 0
    soft_skill_count: int = 0
    hard_matched_count: int = 0
    requirement_score: float = 0.0
    r_weight: float = 1.0
    r_gate: float = 1.0
    v2_final: float = 0.0
    requirement_weight: float = 0.0


@dataclass(frozen=True)
class ScoreInputs:
    raw_semantic: float
    section_raw: float
    skills_coverage: float
    hard_coverage: float
    hard_weight_sum: float
    hard_skill_count: int
    soft_skill_count: int
    hard_matched_count: int
    requirement_scores: tuple[float, ...] = ()


def v2_skill_coverage(
        jd_weights: dict[str, float],
        skill_types: dict[str, str],
        lexical_matches: set[str],
        semantic_matches: set[str],
        evidence_weights: dict[str, float],
        semantic_credit: float,
) -> tuple[float, float, int]:
    total = sum(jd_weights.values())
    hard_total = sum(
        weight for skill, weight in jd_weights.items()
        if skill_types.get(skill, "hard") == "hard"
    )
    covered = 0.0
    hard_covered = 0.0
    hard_matched = 0
    for skill, weight in jd_weights.items():
        if skill in lexical_matches:
            credit = 1.0
        elif skill in semantic_matches:
            credit = semantic_credit
        else:
            credit = 0.0
        evidence = evidence_weights.get(skill, 0.0)
        covered += weight * credit * evidence
        if skill_types.get(skill, "hard") == "hard":
            hard_covered += weight * credit * evidence
            if credit > 0.0:
                hard_matched += 1
    return (
        float(np.clip(
            covered / total if total > 0.0 else 0.0, 0.0, 1.0
        )),
        float(np.clip(
            hard_covered / hard_total if hard_total > 0.0 else 0.0,
            0.0,
            1.0,
        )),
        hard_matched,
    )


def topk_mean(resume_emb: np.ndarray, jd_emb: np.ndarray,
              k: int = 3) -> float:
    if resume_emb.size == 0 or jd_emb.size == 0:
        return 0.0
    sim = resume_emb @ jd_emb.T
    per_chunk = sim.max(axis=1)
    k = min(k, per_chunk.shape[0])
    top = np.sort(per_chunk)[-k:]
    return float(np.clip(top.mean(), 0.0, 1.0))


def calibrate(raw: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((raw - low) / (high - low), 0.0, 1.0))


def load_calibrator(settings: Settings) -> IsotonicRegression | None:
    path = settings.calibrator_path()
    if not path.is_file():
        return None
    return joblib.load(path)


def semantic_probability(
        raw: float,
        settings: Settings,
        calibrator: IsotonicRegression | None,
) -> float:
    if calibrator is None:
        return calibrate(raw, settings.semantic_low, settings.semantic_high)
    predicted = calibrator.predict(np.asarray([[raw]], dtype=np.float64))
    return float(np.clip(predicted[0], 0.0, 1.0))


def skill_coverage(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(np.clip(matched / total, 0.0, 1.0))


def evidence_weighted_coverage(
        jd_weights: dict[str, float],
        evidence_weights: dict[str, float],
) -> float:
    total = sum(jd_weights.values())
    if total <= 0.0:
        return 0.0
    covered = sum(
        weight * evidence_weights.get(skill, 0.0)
        for skill, weight in jd_weights.items()
    )
    return float(np.clip(covered / total, 0.0, 1.0))


def section_fit_score(jd_vec: np.ndarray,
                      section_vec: np.ndarray) -> float:
    if jd_vec.size == 0 or section_vec.size == 0:
        return 0.0
    return float(np.clip(np.dot(jd_vec, section_vec), 0.0, 1.0))


def combine(semantic: float, skills: float, section: float,
            settings: Settings,
            calibrator: IsotonicRegression | None = None) -> ScoreParts:
    return _combine(
        semantic,
        skills,
        section,
        settings,
        settings.semantic_weight,
        settings.skill_weight,
        settings.section_weight,
        calibrator,
    )


def combine_with_signal(
        semantic: float,
        skills: float,
        section: float,
        settings: Settings,
        hard_skill_count: int,
        soft_skill_count: int,
        hard_matched_count: int,
        calibrator: IsotonicRegression | None = None,
) -> ScoreParts:
    limited_signal = (
        hard_skill_count < settings.min_hard_skills
        and hard_skill_count + 0.5 * soft_skill_count
        < settings.min_hard_skills
    )
    semantic_weight = settings.semantic_weight
    skill_weight = settings.skill_weight
    section_weight = settings.section_weight
    if limited_signal:
        remaining = semantic_weight + section_weight
        if remaining > 0:
            semantic_weight /= remaining
            section_weight /= remaining
        else:
            semantic_weight = 1.0
            section_weight = 0.0
        skill_weight = 0.0
    penalty = (
        0.6
        if (
            hard_skill_count >= settings.min_hard_skills
            and hard_matched_count == 0
        )
        else 1.0
    )
    return _combine(
        semantic,
        skills,
        section,
        settings,
        semantic_weight,
        skill_weight,
        section_weight,
        calibrator,
        limited_signal,
        penalty,
    )


def apply_display_map(score: float, spec: str) -> int:
    clamped = float(np.clip(score, 0.0, 1.0))
    if not spec.strip():
        return int(round(clamped * 100))
    pairs = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError("scoring_display_map must contain x:y pairs")
        x, y = (float(value.strip()) for value in parts)
        if not isfinite(x) or not isfinite(y):
            raise ValueError("scoring_display_map must contain finite pairs")
        pairs.append((x, y))
    if clamped <= pairs[0][0]:
        return int(round(pairs[0][1]))
    for (left_x, left_y), (right_x, right_y) in zip(
        pairs, pairs[1:]
    ):
        if clamped <= right_x:
            fraction = (clamped - left_x) / (right_x - left_x)
            return int(round(left_y + fraction * (right_y - left_y)))
    return int(round(pairs[-1][1]))


def combine_v2(
        inputs: ScoreInputs,
        settings: Settings,
        calibrator: IsotonicRegression | None = None,
) -> ScoreParts:
    calibrated = semantic_probability(
        inputs.raw_semantic, settings, calibrator
    )
    semantic = (
        calibrated
        if settings.semantic_source == "calibrated"
        else calibrate(
            inputs.raw_semantic,
            settings.semantic_low,
            settings.semantic_high,
        )
    )
    section = calibrate(
        inputs.section_raw, settings.section_low, settings.section_high
    )
    requirement_values = tuple(
        calibrate(score, settings.requirement_low, settings.requirement_high)
        for score in inputs.requirement_scores
    )
    requirement = (
        float(np.mean(requirement_values))
        if requirement_values else 0.0
    )
    n_eff = inputs.hard_weight_sum + 0.5 * inputs.soft_skill_count
    weight_scale = float(settings.v2_weight_reliability_skills)
    gate_scale = float(settings.v2_gate_reliability_skills)
    r_weight = float(np.clip(n_eff / weight_scale, 0.0, 1.0))
    r_gate = float(np.clip(
        inputs.hard_weight_sum / gate_scale, 0.0, 1.0
    ))
    gate = 1.0 - (
        (1.0 - settings.gate_floor)
        * r_gate
        * (1.0 - inputs.hard_coverage) ** settings.gate_gamma
    )
    req_cfg = (
        settings.skill_weight * settings.unlisted_requirement_share
        if requirement_values else 0.0
    )
    hard_cfg = settings.skill_weight - req_cfg
    hard_weight = hard_cfg * r_weight
    freed = hard_cfg - hard_weight
    active_total = settings.semantic_weight + settings.section_weight
    active_total += req_cfg
    if active_total > 0.0:
        semantic_weight = (
            settings.semantic_weight
            + freed * settings.semantic_weight / active_total
        )
        section_weight = (
            settings.section_weight
            + freed * settings.section_weight / active_total
        )
        requirement_weight = req_cfg + freed * req_cfg / active_total
    else:
        semantic_weight = 1.0
        section_weight = 0.0
        requirement_weight = 0.0
    base = (
        semantic_weight * semantic
        + hard_weight * inputs.skills_coverage
        + section_weight * section
        + requirement_weight * requirement
    )
    final = float(np.clip(base, 0.0, 1.0) * gate)
    return ScoreParts(
        semantic=semantic,
        calibrated_semantic=calibrated,
        skill_coverage=inputs.skills_coverage,
        section_fit=section,
        final=final,
        semantic_display=int(round(semantic * 100)),
        skills_display=int(round(inputs.skills_coverage * 100)),
        section_display=int(round(section * 100)),
        match_percentage=apply_display_map(
            final, settings.scoring_display_map
        ),
        semantic_weight=semantic_weight,
        skill_weight=hard_weight,
        section_weight=section_weight,
        limited_skill_signal=r_weight < 1.0,
        hard_skill_penalty=gate,
        raw_semantic=inputs.raw_semantic,
        section_raw=inputs.section_raw,
        hard_coverage=inputs.hard_coverage,
        hard_weight_sum=inputs.hard_weight_sum,
        hard_skill_count=inputs.hard_skill_count,
        soft_skill_count=inputs.soft_skill_count,
        hard_matched_count=inputs.hard_matched_count,
        requirement_score=requirement,
        r_weight=r_weight,
        r_gate=r_gate,
        v2_final=final,
        requirement_weight=requirement_weight,
    )


def _combine(
        semantic: float,
        skills: float,
        section: float,
        settings: Settings,
        semantic_weight: float,
        skill_weight: float,
        section_weight: float,
        calibrator: IsotonicRegression | None = None,
        limited_signal: bool = False,
        penalty: float = 1.0,
) -> ScoreParts:
    linear_semantic = calibrate(
        semantic, settings.semantic_low, settings.semantic_high
    )
    calibrated = semantic_probability(semantic, settings, calibrator)
    weighted = (
        semantic_weight * linear_semantic
        + skill_weight * skills
        + section_weight * section
    )
    final = float(np.clip(weighted * penalty, 0.0, 1.0))
    sem_i = int(round(linear_semantic * 100))
    sk_i = int(round(skills * 100))
    sec_i = int(round(section * 100))
    return ScoreParts(
        semantic=linear_semantic,
        calibrated_semantic=calibrated,
        skill_coverage=skills,
        section_fit=section,
        final=final,
        semantic_display=sem_i,
        skills_display=sk_i,
        section_display=sec_i,
        match_percentage=int(round(final * 100)),
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
        section_weight=section_weight,
        limited_skill_signal=limited_signal,
        hard_skill_penalty=penalty,
    )
