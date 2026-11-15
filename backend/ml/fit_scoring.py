from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

LABEL_VALUE = {"Weak": 0, "Good": 1, "Great": 2}
SCALE_VALUES = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
GATE_FLOORS = (0.35, 0.5, 0.65, 1.0)
GATE_GAMMAS = (0.5, 1.0, 2.0)
REQUIREMENT_SHARES = (0.0, 0.1, 0.2, 0.3)


def _read_features(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "model_name", "model_source", "jd_id", "label",
        "calibrated_semantic", "section_raw", "skills_coverage",
        "hard_coverage", "hard_weight_sum", "soft_skill_count",
        "hard_matched_count",
        "split", "pool_role",
        "n_req_kept", "n_req_dropped", "legacy_final", "v2_final",
        "n_sections", "has_experience_section",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("features.csv is missing required columns")
    for row in rows:
        row["label_value"] = LABEL_VALUE[row["label"]]
        for field in (
            "calibrated_semantic", "linear_semantic", "section_raw",
            "skills_coverage",
            "hard_coverage", "hard_weight_sum", "soft_skill_count",
            "n_req_kept", "n_req_dropped", "legacy_final", "v2_final",
            "n_sections",
        ):
            row[field] = float(row[field] or 0.0)
        row["has_experience_section"] = (
            str(row["has_experience_section"]).casefold()
            in {"1", "true", "yes"}
        )
        row["requirement_mean"] = _requirement_mean(
            row.get("requirement_scores_summary", "")
        )
    return rows


def _requirement_mean(summary: str) -> float:
    for part in summary.split(";"):
        if part.startswith("mean="):
            return float(part.split("=", 1)[1])
    return 0.0


def _window(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low, high = np.percentile(np.asarray(values), [10.0, 90.0])
    if high <= low:
        return float(low), float(low + 1e-6)
    return float(low), float(high)


def _score(
        row: dict[str, Any],
        params: dict[str, float],
        semantic_field: str = "calibrated_semantic",
) -> float:
    sem = float(np.clip(
        (row[semantic_field] - params["semantic_low"])
        / (params["semantic_high"] - params["semantic_low"]),
        0.0,
        1.0,
    ))
    sec = float(np.clip(
        (row["section_raw"] - params["section_low"])
        / (params["section_high"] - params["section_low"]),
        0.0,
        1.0,
    ))
    req = float(np.clip(
        (row["requirement_mean"] - params["requirement_low"])
        / (params["requirement_high"] - params["requirement_low"]),
        0.0,
        1.0,
    ))
    r_weight = min(
        1.0,
        (row["hard_weight_sum"] + 0.5 * row["soft_skill_count"])
        / params["weight_scale"],
    )
    r_gate = min(1.0, row["hard_weight_sum"] / params["gate_scale"])
    gate = 1.0 - (
        (1.0 - params["gate_floor"])
        * r_gate
        * (1.0 - row["hard_coverage"]) ** params["gate_gamma"]
    )
    hard_weight = params["skill_weight"] * (
        1.0 - params["requirement_share"]
    ) * r_weight
    req_weight_cfg = (
        params["skill_weight"] * params["requirement_share"]
        if row["n_req_kept"] > 0
        else 0.0
    )
    freed = (
        params["skill_weight"] * (1.0 - params["requirement_share"])
        - hard_weight
    )
    active = params["semantic_weight"] + params["section_weight"]
    active += req_weight_cfg
    if active <= 0.0:
        return 0.0
    semantic_weight = (
        params["semantic_weight"]
        + freed * params["semantic_weight"] / active
    )
    section_weight = (
        params["section_weight"]
        + freed * params["section_weight"] / active
    )
    requirement_weight = (
        req_weight_cfg + freed * req_weight_cfg / active
    )
    base = (
        semantic_weight * sem
        + hard_weight * row["skills_coverage"]
        + section_weight * sec
        + requirement_weight * req
    )
    return float(np.clip(base, 0.0, 1.0) * gate)


def _pairwise_loss(
        rows: list[dict[str, Any]],
        scores: list[float],
) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["jd_id"]].append(index)
    losses = []
    for indexes in grouped.values():
        for left in indexes:
            for right in indexes:
                delta = row_delta = (
                    rows[left]["label_value"] - rows[right]["label_value"]
                )
                if row_delta <= 0:
                    continue
                losses.append(
                    math.log1p(math.exp(
                        -max(-40.0, min(40.0, scores[left] - scores[right]))
                    ))
                )
    return float(np.mean(losses)) if losses else 0.0


def _macro_f1(labels: list[int], predicted: list[int]) -> float:
    values = []
    for target in (0, 1, 2):
        tp = sum(a == target and b == target for a, b in zip(labels, predicted))
        fp = sum(a != target and b == target for a, b in zip(labels, predicted))
        fn = sum(a == target and b != target for a, b in zip(labels, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return float(np.mean(values))


def _thresholds(
        labels: list[int],
        scores: list[float],
) -> tuple[float, float, float]:
    candidates = sorted(set(
        [0.0, 1.0]
        + [float(value) for value in scores]
        + list(np.linspace(0.0, 1.0, 21))
    ))
    best = (-1.0, 0.7, 0.85)
    for low in candidates:
        for high in candidates:
            if high <= low:
                continue
            predicted = [
                0 if score < low else 1 if score < high else 2
                for score in scores
            ]
            macro = _macro_f1(labels, predicted)
            if macro > best[0]:
                best = (macro, low, high)
    return best


def _candidate_params(
        rows: list[dict[str, Any]],
        rng: np.random.Generator,
) -> list[dict[str, float]]:
    section_low, section_high = _window(
        [row["section_raw"] for row in rows]
    )
    requirement_low, requirement_high = _window(
        [row["requirement_mean"] for row in rows]
    )
    candidates: list[dict[str, float]] = []
    for _ in range(96):
        weights = rng.dirichlet(np.ones(3))
        candidates.append({
            "semantic_weight": float(weights[0]),
            "skill_weight": float(weights[1]),
            "section_weight": float(weights[2]),
            "gate_floor": float(rng.choice(GATE_FLOORS)),
            "gate_gamma": float(rng.choice(GATE_GAMMAS)),
            "semantic_skill_credit": float(
                rng.choice((0.4, 0.6, 0.8))
            ),
            "requirement_share": float(
                rng.choice(REQUIREMENT_SHARES)
            ),
            "weight_scale": float(rng.choice(SCALE_VALUES)),
            "gate_scale": float(rng.choice(SCALE_VALUES)),
            "semantic_low": 0.0,
            "semantic_high": 1.0,
            "section_low": section_low,
            "section_high": section_high,
            "requirement_low": requirement_low,
            "requirement_high": requirement_high,
        })
    return candidates


def _fit_fold(rows: list[dict[str, Any]]) -> dict[str, float]:
    rng = np.random.default_rng(42)
    best: tuple[float, dict[str, float]] | None = None
    for params in _candidate_params(rows, rng):
        scores = [_score(row, params) for row in rows]
        labels = [row["label_value"] for row in rows]
        _, low, high = _thresholds(labels, scores)
        predicted = [
            0 if score < low else 1 if score < high else 2
            for score in scores
        ]
        ordinal = 1.0 - _macro_f1(labels, predicted)
        objective = _pairwise_loss(rows, scores) + ordinal
        if best is None or objective < best[0]:
            params = dict(params)
            params["threshold_low"] = low
            params["threshold_high"] = high
            best = objective, params
    if best is None:
        raise ValueError("No fitting candidates were generated")
    return best[1]


def _training_rows(
        rows: list[dict[str, Any]],
        held_out_jd: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if row["jd_id"] != held_out_jd]


def _default_params(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "semantic_weight": 0.5,
        "skill_weight": 0.4,
        "section_weight": 0.1,
        "gate_floor": 0.5,
        "gate_gamma": 1.0,
        "semantic_skill_credit": 0.6,
        "requirement_share": 0.2,
        "weight_scale": 4.0,
        "gate_scale": 4.0,
        "semantic_low": 0.0,
        "semantic_high": 1.0,
        "section_low": 0.0,
        "section_high": 1.0,
        "requirement_low": 0.0,
        "requirement_high": 1.0,
    }


def _fit_gate_only(rows: list[dict[str, Any]]) -> dict[str, float]:
    best: tuple[float, dict[str, float]] | None = None
    base = _default_params(rows)
    labels = [row["label_value"] for row in rows]
    for floor in GATE_FLOORS:
        for gamma in GATE_GAMMAS:
            params = dict(base, gate_floor=floor, gate_gamma=gamma)
            scores = [_score(row, params) for row in rows]
            _, low, high = _thresholds(labels, scores)
            predicted = [
                0 if score < low else 1 if score < high else 2
                for score in scores
            ]
            objective = _pairwise_loss(rows, scores) + (
                1.0 - _macro_f1(labels, predicted)
            )
            if best is None or objective < best[0]:
                best = objective, dict(
                    params,
                    threshold_low=low,
                    threshold_high=high,
                )
    if best is None:
        raise ValueError("No gate-only candidates were generated")
    return best[1]


def _fit_thresholds(
        rows: list[dict[str, Any]],
        score_fn: Any,
) -> tuple[float, float]:
    labels = [row["label_value"] for row in rows]
    scores = [score_fn(row) for row in rows]
    _, low, high = _thresholds(labels, scores)
    return low, high


def _nested_variants(
        rows: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["jd_id"]].append(row)
    names = ("a_legacy", "b_calibrated", "c_window", "d_gate_only", "e_full")
    predictions = {name: [0.0] * len(rows) for name in names}
    fold_params: dict[str, dict[str, Any]] = {}
    for held_out in grouped:
        train = _training_rows(rows, held_out)
        full = _fit_fold(train)
        gate = _fit_gate_only(train)
        default = _default_params(train)
        legacy_low, legacy_high = _fit_thresholds(
            train, lambda row: float(row["legacy_final"])
        )
        calibrated_low, calibrated_high = _fit_thresholds(
            train, lambda row: _score(row, default, "calibrated_semantic")
        )
        window_low, window_high = _fit_thresholds(
            train, lambda row: _score(row, default, "linear_semantic")
        )
        gate_low, gate_high = gate["threshold_low"], gate["threshold_high"]
        fold_params[held_out] = {
            "legacy": (legacy_low, legacy_high),
            "calibrated": (calibrated_low, calibrated_high),
            "window": (window_low, window_high),
            "gate_only": gate,
            "full": full,
        }
        for index, row in enumerate(rows):
            if row["jd_id"] != held_out:
                continue
            predictions["a_legacy"][index] = float(row["legacy_final"])
            predictions["b_calibrated"][index] = _score(
                row, default, "calibrated_semantic"
            )
            predictions["c_window"][index] = _score(
                row, default, "linear_semantic"
            )
            predictions["d_gate_only"][index] = _score(row, gate)
            predictions["e_full"][index] = _score(row, full)
    return predictions, fold_params


def _macro_f1_from_scores(
        rows: list[dict[str, Any]],
        scores: list[float],
        low: float,
        high: float,
) -> float:
    labels = [row["label_value"] for row in rows]
    predicted = [0 if s < low else 1 if s < high else 2 for s in scores]
    return _macro_f1(labels, predicted)


def _metric_values(
        rows: list[dict[str, Any]],
        scores: list[float],
        low: float,
        high: float,
) -> dict[str, float]:
    labels = [row["label_value"] for row in rows]
    return {
        "pairwise": _pairwise_accuracy(rows, scores),
        "great_auc": _auc(labels, scores, 2),
        "good_weak_auc": _auc(labels, scores, 1),
        "macro_f1": _macro_f1_from_scores(rows, scores, low, high),
    }


def _hierarchical_bootstrap(
        rows: list[dict[str, Any]],
        score_map: dict[str, list[float]],
        metric: str,
        variant_a: str,
        variant_b: str | None,
        thresholds: dict[str, tuple[float, float]],
        seed: int = 42,
        n_resamples: int = 1000,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["pool_role"] != "in_category":
            continue
        grouped[row["jd_id"]].append(index)
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    jds = list(grouped)
    values = []
    for _ in range(n_resamples):
        sampled: list[int] = []
        for jd_index in rng.integers(0, len(jds), len(jds)):
            indexes = grouped[jds[int(jd_index)]]
            sampled.extend(
                int(value) for value in rng.choice(indexes, len(indexes))
            )
        boot_rows = [rows[index] for index in sampled]
        a_scores = [score_map[variant_a][index] for index in sampled]
        if metric == "macro_f1":
            a_value = _metric_values(
                boot_rows, a_scores, *thresholds[variant_a]
            )[metric]
        else:
            a_value = _metric_values(
                boot_rows, a_scores, *thresholds[variant_a]
            )[metric]
        if variant_b is not None:
            b_scores = [score_map[variant_b][index] for index in sampled]
            b_value = _metric_values(
                boot_rows, b_scores, *thresholds[variant_b]
            )[metric]
            values.append(a_value - b_value)
        else:
            values.append(a_value)
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def _format_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _comprehensive_fit(
        features_path: Path,
        report_path: Path,
) -> dict[str, Any]:
    rows = _read_features(features_path)
    model_name = rows[0]["model_name"] if rows else "unknown"
    model_source = rows[0]["model_source"] if rows else "unknown"
    jd_count = len({row["jd_id"] for row in rows})
    if len(rows) < 60 or jd_count < 5:
        missing = []
        if len(rows) < 60:
            missing.append(f"at least 60 labeled pairs (found {len(rows)})")
        if jd_count < 5:
            missing.append(f"at least 5 JDs (found {jd_count})")
        message = "INSUFFICIENT_DATA: missing " + "; ".join(missing) + "."
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            f"# Scoring fit report: model={model_name}; "
            f"source={model_source}\n\n{message}\n",
            encoding="utf-8",
        )
        print(message)
        return {"status": "insufficient", "missing": missing}
    predictions, fold_params = _nested_variants(rows)
    in_rows = [
        (index, row) for index, row in enumerate(rows)
        if row["pool_role"] == "in_category"
    ]
    eval_rows = [row for _, row in in_rows]
    eval_indices = [index for index, _ in in_rows]
    eval_scores = {
        name: [values[index] for index in eval_indices]
        for name, values in predictions.items()
    }
    thresholds = {
        "a_legacy": (
            statistics.median(v["legacy"][0] for v in fold_params.values()),
            statistics.median(v["legacy"][1] for v in fold_params.values()),
        ),
        "b_calibrated": (
            statistics.median(v["calibrated"][0] for v in fold_params.values()),
            statistics.median(v["calibrated"][1] for v in fold_params.values()),
        ),
        "c_window": (
            statistics.median(v["window"][0] for v in fold_params.values()),
            statistics.median(v["window"][1] for v in fold_params.values()),
        ),
        "d_gate_only": (
            statistics.median(v["gate_only"]["threshold_low"] for v in fold_params.values()),
            statistics.median(v["gate_only"]["threshold_high"] for v in fold_params.values()),
        ),
        "e_full": (
            statistics.median(v["full"]["threshold_low"] for v in fold_params.values()),
            statistics.median(v["full"]["threshold_high"] for v in fold_params.values()),
        ),
    }
    labels = [row["label_value"] for row in eval_rows]
    lines = [
        f"# Scoring fit report: model={model_name}; source={model_source}",
        "",
        "Nested leave-one-JD-out evaluation; all score thresholds are fitted "
        "inside the training fold. Bootstrap CIs use 1,000 hierarchical "
        "JD-and-row resamples over `pool_role=in_category`.",
        "",
        f"- rows: {len(rows)}; JDs: {len({r['jd_id'] for r in rows})}",
        "",
        "## Headline in_category metrics",
        "",
        "| Variant | Pairwise | Great-v-rest AUC | Good-v-Weak AUC | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics: dict[str, dict[str, float]] = {}
    for name in predictions:
        values = _metric_values(
            eval_rows, eval_scores[name], *thresholds[name]
        )
        metrics[name] = values
        lines.append(
            f"| {name} | {values['pairwise']:.3f} | "
            f"{values['great_auc']:.3f} | {values['good_weak_auc']:.3f} | "
            f"{values['macro_f1']:.3f} |"
        )
    baseline_scores = {
        "raw_semantic": [float(row["raw_semantic"]) for row in rows],
        "skills_coverage": [float(row["skills_coverage"]) for row in rows],
        "hard_coverage": [float(row["hard_coverage"]) for row in rows],
    }
    baseline_thresholds: dict[str, tuple[float, float]] = {}
    for name in baseline_scores:
        fold_values = []
        for jd in sorted({row["jd_id"] for row in rows}):
            train = _training_rows(rows, jd)
            low, high = _fit_thresholds(
                train,
                lambda row, key=name: float(row[key]),
            )
            fold_values.append((low, high))
        baseline_thresholds[name] = (
            statistics.median(value[0] for value in fold_values),
            statistics.median(value[1] for value in fold_values),
        )
    lines.extend(["", "## Baselines", ""])
    for name, values in baseline_scores.items():
        subset = [values[index] for index in eval_indices]
        baseline_metric = _metric_values(
            eval_rows, subset, *baseline_thresholds[name]
        )
        lines.append(
            f"- {name}: pairwise={baseline_metric['pairwise']:.3f}, "
            f"Great-v-rest AUC={baseline_metric['great_auc']:.3f}, "
            f"Good-v-Weak AUC={baseline_metric['good_weak_auc']:.3f}, "
            f"macro-F1={baseline_metric['macro_f1']:.3f}"
        )
    lines.extend(["", "## Hierarchical bootstrap 95% CIs", ""])
    for metric in ("pairwise", "great_auc", "good_weak_auc", "macro_f1"):
        ci = _hierarchical_bootstrap(
            eval_rows, eval_scores, metric, "b_calibrated", None,
            thresholds
        )
        lines.append(f"- b calibrated {metric}: {_format_ci(ci)}")
        for left, right, label in (
            ("b_calibrated", "a_legacy", "b-a"),
            ("c_window", "b_calibrated", "c-b"),
            ("d_gate_only", "b_calibrated", "d-b"),
            ("e_full", "b_calibrated", "e-b"),
        ):
            ci = _hierarchical_bootstrap(
                eval_rows, eval_scores, metric, left, right, thresholds
            )
            lines.append(f"- {label} {metric}: {_format_ci(ci)}")
    lines.extend(["", "## Calibrated vs window (split != val)", ""])
    non_val = [
        index for index, row in enumerate(rows)
        if row["pool_role"] == "in_category" and row["split"] != "val"
    ]
    nv_rows = [rows[index] for index in non_val]
    for name in ("b_calibrated", "c_window"):
        values = _metric_values(
            nv_rows, [predictions[name][i] for i in non_val],
            *thresholds[name]
        )
        lines.append(f"- {name}: {values}")
    lines.extend(["", "## Required slices", ""])
    slice_specs = {
        "pool_role=in_category": lambda r: r["pool_role"] == "in_category",
        "r_weight<1": lambda r: min(
            1.0, (r["hard_weight_sum"] + 0.5 * r["soft_skill_count"]) / 4.0
        ) < 1.0,
        "has_experience_section": lambda r: r["has_experience_section"],
        "zero-match": lambda r: r["hard_matched_count"] == 0,
        "no-hard-skill JDs": lambda r: r["jd_id"] in {
            "handwritten_00", "handwritten_02", "handwritten_03"
        },
    }
    for label, predicate in slice_specs.items():
        indexes = [
            i for i, row in enumerate(rows)
            if row["pool_role"] == "in_category" and predicate(row)
        ]
        subset = [rows[i] for i in indexes]
        scores = [predictions["b_calibrated"][i] for i in indexes]
        if subset:
            lines.append(
                f"- {label} (n={len(subset)}): "
                f"{_metric_values(subset, scores, *thresholds['b_calibrated'])}"
            )
        else:
            lines.append(f"- {label}: unavailable (n=0)")
    lines.extend(["", "## ID-only disagreements", ""])
    for left, right in (
        ("a_legacy", "b_calibrated"), ("b_calibrated", "c_window"),
        ("b_calibrated", "d_gate_only"), ("b_calibrated", "e_full"),
    ):
        ids = [
            rows[i]["resume_id"] for i in eval_indices
            if (
                (0 if eval_scores[left][eval_indices.index(i)] < thresholds[left][0]
                 else 1 if eval_scores[left][eval_indices.index(i)] < thresholds[left][1] else 2)
                !=
                (0 if eval_scores[right][eval_indices.index(i)] < thresholds[right][0]
                 else 1 if eval_scores[right][eval_indices.index(i)] < thresholds[right][1] else 2)
            )
        ]
        lines.append(f"- {left} vs {right}: {ids}")
    lines.extend(["", "## Nested ablations and parameter stability", ""])
    for name in ("d_gate_only", "e_full"):
        lines.append(f"- {name}: thresholds={thresholds[name]}")
    for key in ("semantic_weight", "skill_weight", "section_weight",
                "gate_floor", "gate_gamma", "requirement_share",
                "weight_scale", "gate_scale"):
        values = [v["full"][key] for v in fold_params.values()]
        lines.append(
            f"- full {key}: median={statistics.median(values):.4f}, "
            f"range=[{min(values):.4f}, {max(values):.4f}]"
        )
    lines.extend(["", "## Nested macro-F1 cutoffs", ""])
    for jd, values in fold_params.items():
        lines.append(
            f"- {jd}: legacy={values['legacy']}, calibrated={values['calibrated']}, "
            f"window={values['window']}, gate_only="
            f"{(values['gate_only']['threshold_low'], values['gate_only']['threshold_high'])}, "
            f"full={(values['full']['threshold_low'], values['full']['threshold_high'])}"
        )
    low, high = thresholds["b_calibrated"]
    if high <= low:
        high = low + 0.01
    lines.extend([
        "",
        "## Display map recommendation",
        "",
        f"Strictly increasing knots: score=0 -> 0, score={low:.6f} -> 70, "
        f"score={high:.6f} -> 85, score=1 -> 100.",
        "",
        "Proposed `.env` lines (text only; not applied):",
        "SCORING_DISPLAY_MAP=0:0,"
        f"{low:.6f}:70,{high:.6f}:85,1:100",
        "SCORING_MODE=v2",
        "SCORING_SEMANTIC_SOURCE=calibrated",
        "",
        "## Recommendation",
        "",
        "Use the nested winner only as an evaluation result; do not promote "
        "weights or cutoffs without a fresh holdout. The labeled pool is "
        "small and JD-correlated, so fold selection and parameter ranges "
        "can overfit despite the nested protocol.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return {"status": "fit", "metrics": metrics, "fold_params": fold_params}


def _nested_fold_predictions(
        rows: list[dict[str, Any]],
) -> tuple[list[float], dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["jd_id"]].append(row)
    predictions = [0.0] * len(rows)
    fold_params: dict[str, dict[str, float]] = {}
    for held_out in grouped:
        params = _fit_fold(_training_rows(rows, held_out))
        fold_params[held_out] = params
        for index, row in enumerate(rows):
            if row["jd_id"] == held_out:
                predictions[index] = _score(row, params)
    return predictions, fold_params


def _auc(labels: list[int], scores: list[float], positive: int) -> float:
    binary = [int(label == positive) for label in labels]
    if len(set(binary)) < 2:
        return float("nan")
    return float(roc_auc_score(binary, scores))


def _pairwise_accuracy(rows: list[dict[str, Any]], scores: list[float]) -> float:
    total = correct = 0
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["jd_id"]].append(index)
    for indexes in grouped.values():
        for left in indexes:
            for right in indexes:
                if rows[left]["label_value"] <= rows[right]["label_value"]:
                    continue
                total += 1
                correct += scores[left] > scores[right]
    return correct / total if total else float("nan")


def _ndcg(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    output = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["jd_id"]].append(index)
    for jd_id, indexes in grouped.items():
        order = sorted(indexes, key=lambda index: scores[index], reverse=True)
        ideal = sorted(
            (rows[index]["label_value"] for index in indexes),
            reverse=True,
        )
        dcg = sum(
            (2 ** rows[index]["label_value"] - 1) / math.log2(rank + 2)
            for rank, index in enumerate(order)
        )
        ideal_dcg = sum(
            (2 ** value - 1) / math.log2(rank + 2)
            for rank, value in enumerate(ideal)
        )
        output[jd_id] = dcg / ideal_dcg if ideal_dcg else 1.0
    return output


def _confusion(
        labels: list[int],
        scores: list[float],
        low: float,
        high: float,
) -> list[list[int]]:
    matrix = [[0, 0, 0] for _ in range(3)]
    for label, score in zip(labels, scores):
        predicted = 0 if score < low else 1 if score < high else 2
        matrix[label][predicted] += 1
    return matrix


def _report_metrics(
        rows: list[dict[str, Any]],
        scores: list[float],
        label: str,
        thresholds: tuple[float, float],
) -> list[str]:
    labels = [row["label_value"] for row in rows]
    low, high = thresholds
    distinct: dict[str, int] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["jd_id"]].append(score)
    for jd_id, values in grouped.items():
        distinct[jd_id] = len({round(value * 100) for value in values})
    return [
        f"- {label} pairwise accuracy: "
        f"{_pairwise_accuracy(rows, scores):.4f}",
        f"- {label} Great-v-rest AUC: "
        f"{_auc(labels, scores, 2):.4f}",
        f"- {label} Good-v-Weak AUC: "
        f"{_auc(labels, scores, 1):.4f}",
        f"- {label} share >= 90: "
        f"{sum(score >= .9 for score in scores) / len(scores):.4f}",
        f"- {label} mean distinct scores/JD: "
        f"{np.mean(list(distinct.values())):.2f}",
        f"- {label} confusion: {_confusion(labels, scores, low, high)}",
    ]


def fit_features(
        features_path: Path,
        report_path: Path,
) -> dict[str, Any]:
    return _comprehensive_fit(features_path, report_path)
    # Kept below as historical reference for the earlier single-variant
    # report format.
    rows = _read_features(features_path)
    model_name = rows[0]["model_name"] if rows else "unknown"
    model_source = rows[0]["model_source"] if rows else "unknown"
    jd_count = len({row["jd_id"] for row in rows})
    missing: list[str] = []
    if len(rows) < 60:
        missing.append(
            f"at least 60 labeled pairs (found {len(rows)})"
        )
    if jd_count < 5:
        missing.append(f"at least 5 JDs (found {jd_count})")
    report_lines = [
        f"# Scoring fit report: model={model_name}; source={model_source}",
        "",
    ]
    if missing:
        message = "INSUFFICIENT_DATA: missing " + "; ".join(missing) + "."
        print(message)
        print(
            "Refusing to fit weights, semantic credit, requirement share, "
            "windows, or reliability scales; only gate_floor and gate_gamma "
            "may be fitted after sufficient data."
        )
        report_lines.extend([
            message,
            "",
            "No fitted values were produced. Gate parameters were not "
            "written to configuration.",
        ])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return {"status": "insufficient", "missing": missing}

    predictions, fold_params = _nested_fold_predictions(rows)
    params = next(iter(fold_params.values()))
    labels = [row["label_value"] for row in rows]
    thresholds = (params["threshold_low"], params["threshold_high"])
    threshold_summary = {
        jd: (values["threshold_low"], values["threshold_high"])
        for jd, values in fold_params.items()
    }
    report_lines.extend([
        f"- rows: {len(rows)}",
        f"- JDs: {jd_count}",
        f"- fitted parameters: {params}",
        f"- held-out thresholds by JD: {threshold_summary}",
        "",
        "## Metrics",
        "",
    ])
    report_lines.extend(
        _report_metrics(rows, predictions, "v2", thresholds)
    )
    in_domain = [
        (row, score) for row, score in zip(rows, predictions)
        if row["pool_role"] == "in_category"
    ]
    sample_domain = [
        (row, score) for row, score in zip(rows, predictions)
        if row["pool_role"] == "sample"
    ]
    if in_domain:
        in_auc = _auc(
            [row["label_value"] for row, _ in in_domain],
            [score for _, score in in_domain],
            2,
        )
        report_lines.append(
            "- Headline within-domain (in_category) Great-v-rest AUC: "
            f"{in_auc:.4f}"
        )
    if sample_domain:
        sample_auc = _auc(
            [row["label_value"] for row, _ in sample_domain],
            [score for _, score in sample_domain],
            2,
        )
        report_lines.append(
            "- Sample-row Great-v-rest AUC (reported separately): "
            f"{sample_auc:.4f}"
        )
    combined_domain = in_domain + sample_domain
    if combined_domain:
        combined_auc = _auc(
            [row["label_value"] for row, _ in combined_domain],
            [score for _, score in combined_domain],
            2,
        )
        report_lines.append(
            "- Combined in_category+sample AUC (footnote): "
            f"{combined_auc:.4f}"
        )
    report_lines.extend([
        "",
        "## Component AUC and ablations",
        "",
    ])
    for name, values in (
        ("semantic", [row["calibrated_semantic"] for row in rows]),
        ("skills", [row["skills_coverage"] for row in rows]),
        ("hard skills", [row["hard_coverage"] for row in rows]),
        ("section fit", [row["section_raw"] for row in rows]),
        ("requirements", [row["requirement_mean"] for row in rows]),
    ):
        report_lines.append(
            f"- {name} Great-v-rest AUC: "
            f"{_auc(labels, values, 2):.4f}"
        )
    report_lines.extend([
        "",
        "## Legacy comparison",
        "",
    ])
    legacy = [row["legacy_final"] for row in rows]
    report_lines.extend(_report_metrics(rows, legacy, "legacy", (0.7, 0.85)))
    report_lines.extend([
        "",
        "## Required slices",
        "",
        "- Zero-hard-match rows: "
        f"{sum(row['hard_matched_count'] == 0 for row in rows)}",
        "- JDs with r_weight < 1: reported by fitted feature values "
        "when available; source rows do not contain per-row r_weight.",
        "- Requirement noise: "
        f"{sum(row['n_req_dropped'] for row in rows)} dropped phrases over "
        f"{sum(row['n_req_kept'] + row['n_req_dropped'] for row in rows)} "
        "extracted phrases.",
        "- Kept-phrase examples are unavailable because features.csv "
        "intentionally contains no JD text.",
        "- The 85/70 cutoffs require human-label validation; this report "
        "uses held-out thresholds rather than changing defaults.",
    ])
    for label, predicate in (
        ("pool_role=in_category", lambda row: row["pool_role"] == "in_category"),
        ("pool_role=other_category", lambda row: row["pool_role"] == "other_category"),
        ("pool_role=sample", lambda row: row["pool_role"] == "sample"),
        ("has_experience_section=true", lambda row: row["has_experience_section"]),
        ("has_experience_section=false", lambda row: not row["has_experience_section"]),
    ):
        subset = [
            (row, score) for row, score in zip(rows, predictions)
            if predicate(row)
        ]
        if subset:
            slice_auc = _auc(
                [row["label_value"] for row, _ in subset],
                [score for _, score in subset],
                2,
            )
            report_lines.append(
                f"- Slice {label} Great-v-rest AUC: "
                f"{slice_auc:.4f} "
                f"(n={len(subset)})."
            )
    r_weight_rows = [
        (row, score) for row, score in zip(rows, predictions)
        if min(1.0, (
            row["hard_weight_sum"] + 0.5 * row["soft_skill_count"]
        ) / params["weight_scale"]) < 1.0
    ]
    if r_weight_rows:
        r_weight_auc = _auc(
            [row["label_value"] for row, _ in r_weight_rows],
            [score for _, score in r_weight_rows],
            2,
        )
        report_lines.append(
            "- Slice r_weight<1 Great-v-rest AUC: "
            f"{r_weight_auc:.4f} (n={len(r_weight_rows)})."
        )
    else:
        report_lines.append(
            "- Slice r_weight<1 Great-v-rest AUC: unavailable (no rows)."
        )
    test_rows = [
        row for row in rows
        if row["split"] in {"test", "sample"}
    ]
    for domain in ("in_category", "sample"):
        domain_rows = [
            row for row in test_rows
            if row["pool_role"] == domain
        ]
        if not domain_rows:
            report_lines.append(
                f"- Test-only {domain} AUC: unavailable (no rows)."
            )
            continue
        domain_labels = [row["label_value"] for row in domain_rows]
        calibrated_scores = [
            _score(row, params, "calibrated_semantic")
            for row in domain_rows
        ]
        window_scores = [
            _score(row, params, "linear_semantic")
            for row in domain_rows
        ]
        report_lines.append(
            f"- Test-only {domain} AUC calibrated/window: "
            f"{_auc(domain_labels, calibrated_scores, 2):.4f}/"
            f"{_auc(domain_labels, window_scores, 2):.4f}."
        )
    combined_test = [
        row for row in test_rows
        if row["pool_role"] in {"in_category", "sample"}
    ]
    if combined_test:
        combined_labels = [row["label_value"] for row in combined_test]
        combined_scores = [
            _score(row, params, "calibrated_semantic")
            for row in combined_test
        ]
        report_lines.append(
            "- Test-only combined in-category+sample AUC: "
            f"{_auc(combined_labels, combined_scores, 2):.4f}."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "fit",
        "params": params,
        "thresholds": thresholds,
        "predictions": predictions,
        "ndcg": _ndcg(rows, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("ml/artifacts/SCORING_FIT_REPORT.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_features(args.features, args.report)


if __name__ == "__main__":
    main()
