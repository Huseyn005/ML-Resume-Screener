from __future__ import annotations

import csv

from ml.collect_features import (
    FEATURE_FIELDS,
    _pool_metadata,
    _resolve_resumes,
    _read_labels,
    write_features,
)
from ml.fit_scoring import fit_features
from ml.fit_scoring import (
    GATE_FLOORS,
    GATE_GAMMAS,
    REQUIREMENT_SHARES,
    SCALE_VALUES,
)


def _feature_rows() -> list[dict[str, object]]:
    rows = []
    for jd_index in range(5):
        for resume_index in range(12):
            value = resume_index / 11
            rows.append({
                "model_name": "synthetic",
                "model_source": "test",
                "jd_id": f"jd_{jd_index}",
                "resume_id": f"r_{resume_index}",
                "split": "test",
                "pool_role": "in_category",
                "label": (
                    "Weak" if resume_index < 4
                    else "Good" if resume_index < 8
                    else "Great"
                ),
                "raw_semantic": value,
                "calibrated_semantic": value,
                "linear_semantic": value,
                "section_raw": value,
                "skills_coverage": value,
                "hard_coverage": value,
                "hard_weight_sum": 4.0,
                "soft_skill_count": 0,
                "hard_matched_count": resume_index,
                "n_semantic_only_matches": 0,
                "n_req_kept": 1,
                "n_req_dropped": 0,
                "requirement_scores_summary": (
                    f"n=1;mean={value};max={value}"
                ),
                "legacy_final": value,
                "v2_final": value,
                "n_sections": 3,
                "has_experience_section": True,
            })
    return rows


def test_synthetic_collect_to_fit_flow(tmp_path) -> None:
    features_path = tmp_path / "features.csv"
    report_path = tmp_path / "report.md"
    write_features(features_path, _feature_rows())

    with features_path.open(encoding="utf-8", newline="") as stream:
        assert next(csv.DictReader(stream)).keys() == set(FEATURE_FIELDS)

    result = fit_features(features_path, report_path)

    assert result["status"] == "fit"
    assert report_path.exists()
    assert "model=synthetic; source=test" in report_path.read_text(
        encoding="utf-8"
    )


def test_fit_refuses_insufficient_data(tmp_path, capsys) -> None:
    features_path = tmp_path / "features.csv"
    report_path = tmp_path / "report.md"
    write_features(features_path, _feature_rows()[:4])

    result = fit_features(features_path, report_path)

    assert result["status"] == "insufficient"
    assert "INSUFFICIENT_DATA" in capsys.readouterr().out


def test_collect_resolves_sample_and_pool_ids(tmp_path) -> None:
    sample_dir = tmp_path / "samples"
    pool_dir = tmp_path / "pool"
    sample_dir.mkdir()
    pool_dir.mkdir()
    (sample_dir / "sample.pdf").write_bytes(b"sample")
    (pool_dir / "p_1234567890.pdf").write_bytes(b"pool")

    mapping = _resolve_resumes(sample_dir, pool_dir)

    assert "sample.pdf" in mapping
    assert "p_1234567890" in mapping


def test_fit_grid_includes_thin_jd_values() -> None:
    assert SCALE_VALUES == (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    assert 1.0 in GATE_FLOORS
    assert GATE_GAMMAS == (0.5, 1.0, 2.0)
    assert 0.0 in REQUIREMENT_SHARES


def test_blank_labels_are_skipped(tmp_path, capsys) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "jd_id,resume_file,label\njd,r1,\njd,r2,Great\n",
        encoding="utf-8",
    )

    rows = _read_labels(labels)

    assert len(rows) == 1
    assert "Skipped 1 blank-label rows" in capsys.readouterr().out


def test_leave_one_jd_out_training_rows() -> None:
    from ml.fit_scoring import _training_rows

    rows = [
        {"jd_id": "jd_a"},
        {"jd_id": "jd_b"},
        {"jd_id": "jd_a"},
    ]

    assert [row["jd_id"] for row in _training_rows(rows, "jd_a")] == ["jd_b"]


def test_manifest_roles_are_per_jd_pair(tmp_path) -> None:
    manifest = tmp_path / "pool_manifest.local.csv"
    manifest.write_text(
        "jd_id,id,split,pool_role\n"
        "jd_a,p_same,test,in_category\n"
        "jd_b,p_same,test,other_category\n",
        encoding="utf-8",
    )

    metadata = _pool_metadata(manifest)

    assert metadata[("jd_a", "p_same")] == ("test", "in_category")
    assert metadata[("jd_b", "p_same")] == ("test", "other_category")


def test_nested_fold_fit_excludes_held_out_jd(monkeypatch) -> None:
    import ml.fit_scoring as fitting

    rows = [
        {"jd_id": "jd_a", "label_value": 0},
        {"jd_id": "jd_a", "label_value": 1},
        {"jd_id": "jd_b", "label_value": 0},
        {"jd_id": "jd_b", "label_value": 1},
    ]
    seen: list[set[str]] = []

    def fake_fit(train):
        seen.append({row["jd_id"] for row in train})
        return {
            "threshold_low": 0.3,
            "threshold_high": 0.7,
        }

    monkeypatch.setattr(fitting, "_fit_fold", fake_fit)
    monkeypatch.setattr(
        fitting,
        "_score",
        lambda row, params: float(row["label_value"]),
    )

    fitting._nested_fold_predictions(rows)

    assert seen == [{"jd_b"}, {"jd_a"}]
