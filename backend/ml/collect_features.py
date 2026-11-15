from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.extract import extract_pdf
from app.preprocess import scrub_pii, split_sections

FEATURE_FIELDS = [
    "model_name",
    "model_source",
    "jd_id",
    "resume_id",
    "split",
    "pool_role",
    "label",
    "raw_semantic",
    "calibrated_semantic",
    "linear_semantic",
    "section_raw",
    "skills_coverage",
    "hard_coverage",
    "hard_weight_sum",
    "soft_skill_count",
    "hard_matched_count",
    "n_semantic_only_matches",
    "n_req_kept",
    "n_req_dropped",
    "requirement_scores_summary",
    "n_sections",
    "has_experience_section",
    "legacy_final",
    "v2_final",
]


def _resume_id(path: Path, index: int) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    return f"r{index:03d}_{digest}"


def _resolve_resumes(
        resumes_dir: Path,
        pool_dir: Path | None = None,
) -> dict[str, Path]:
    paths = [
        path for path in sorted(resumes_dir.glob("*.pdf"))
        if path.name not in {"blank_scanned.pdf", "hidden_text.pdf"}
    ]
    mapping: dict[str, Path] = {}
    for index, path in enumerate(paths):
        identifier = _resume_id(path, index)
        mapping[identifier] = path
        mapping[path.name] = path
        mapping[f"resume_{index:04d}"] = path
    pool_dir = pool_dir or Path(__file__).parent / "artifacts" / "labeling_pool"
    if pool_dir.is_dir():
        for path in sorted(pool_dir.glob("p_*.pdf")):
            mapping[path.stem] = path
            mapping[path.name] = path
    return mapping


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"jd_id", "resume_file", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "labels.csv must contain jd_id,resume_file,label columns"
        )
    allowed = {"Great", "Good", "Weak"}
    skipped = 0
    valid_rows = []
    for row in rows:
        if not row["label"].strip():
            skipped += 1
            continue
        if row["label"] not in allowed:
            raise ValueError("labels must be Great, Good, or Weak")
        valid_rows.append(row)
    print(f"Skipped {skipped} blank-label rows")
    return valid_rows


def _metadata_for_path(path: Path) -> tuple[str, str]:
    if path.parent.name == "labeling_pool":
        return "test", "other_category"
    if path.parent.name == "samples":
        return "sample", "sample"
    return "val", "other_category"


def _pool_metadata(
        manifest_path: Path | None = None,
) -> dict[tuple[str, str], tuple[str, str]]:
    path = manifest_path or (
        Path(__file__).parent / "artifacts" / "pool_manifest.local.csv"
    )
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (row["jd_id"], row["id"]): (row["split"], row["pool_role"])
            for row in csv.DictReader(stream)
        }


def _requirement_summary(detail: dict[str, Any]) -> str:
    scores = [
        float(item["similarity"])
        for item in detail.get("unlisted_requirements", [])
    ]
    if not scores:
        return "n=0;mean=0.000000;max=0.000000"
    return (
        f"n={len(scores)};mean={sum(scores) / len(scores):.6f};"
        f"max={max(scores):.6f}"
    )


def collect_feature_rows(
        labels_path: Path,
        jd_texts: dict[str, str],
        resume_map: dict[str, Path],
        settings: Settings,
) -> list[dict[str, Any]]:
    labels = _read_labels(labels_path)
    original_get_settings = main.get_settings
    rows: list[dict[str, Any]] = []
    pool_metadata = _pool_metadata()
    try:
        main.get_settings = lambda: settings
        with TestClient(main.app) as client:
            health = client.get("/health").json()
            model_name = Path(health["model"]).name
            model_source = health.get("model_source") or "unknown"
            resume_ids = {
                path: _resume_id(path, index)
                for index, path in enumerate(
                    sorted(set(resume_map.values()), key=str)
                )
            }
            for label_row in labels:
                jd_id = label_row["jd_id"]
                resume_key = label_row["resume_file"]
                if jd_id not in jd_texts:
                    raise ValueError(f"Unknown jd_id: {jd_id}")
                if resume_key not in resume_map:
                    raise ValueError(f"Unknown resume_file: {resume_key}")
                path = resume_map[resume_key]
                split, pool_role = pool_metadata.get(
                    (jd_id, resume_key),
                    _metadata_for_path(path),
                )
                extracted = extract_pdf(
                    path.read_bytes(), min_chars=1
                ).text
                sections = split_sections(scrub_pii(extracted))
                with path.open("rb") as stream:
                    response = client.post(
                        "/analyze?debug=true",
                        data={"job_description": jd_texts[jd_id]},
                        files={
                            "resumes": (
                                "resume.pdf", stream, "application/pdf"
                            )
                        },
                    )
                response.raise_for_status()
                detail = response.json()["debug"][0]
                formula = detail["formula"]
                components = detail.get("v2_components", {})
                rows.append({
                    "model_name": model_name,
                    "model_source": model_source,
                    "jd_id": jd_id,
                    "resume_id": (
                        resume_key
                        if resume_key.startswith(("p_", "r"))
                        else resume_ids[path]
                    ),
                    "split": split,
                    "pool_role": pool_role,
                    "label": label_row["label"],
                    "raw_semantic": detail["raw_semantic"],
                    "calibrated_semantic": detail[
                        "calibrated_semantic"
                    ],
                    "linear_semantic": detail["linear_semantic"],
                    "section_raw": detail.get(
                        "section_raw", detail["section_fit"]
                    ),
                    "skills_coverage": detail["skills_coverage"],
                    "hard_coverage": components.get(
                        "hard_coverage", 0.0
                    ),
                    "hard_weight_sum": components.get(
                        "hard_weight_sum", 0.0
                    ),
                    "soft_skill_count": components.get(
                        "soft_skill_count", 0
                    ),
                    "hard_matched_count": components.get(
                        "hard_matched_count", 0
                    ),
                    "n_semantic_only_matches": components.get(
                        "n_semantic_only_matches", 0
                    ),
                    "n_req_kept": components.get(
                        "kept_requirement_count", 0
                    ),
                    "n_req_dropped": components.get(
                        "dropped_requirement_count", 0
                    ),
                    "requirement_scores_summary": _requirement_summary(
                        detail
                    ),
                    "n_sections": sum(
                        bool(value.strip()) for value in sections.values()
                    ),
                    "has_experience_section": bool(
                        sections["experience"].strip()
                    ),
                    "legacy_final": formula.get(
                        "legacy_final", formula["final_score"]
                    ),
                    "v2_final": formula.get(
                        "v2_final", formula["final_score"]
                    ),
                })
    finally:
        main.get_settings = original_get_settings
    return rows


def write_features(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No labeled feature rows were collected")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/artifacts/labels.csv"),
    )
    parser.add_argument(
        "--resumes-dir",
        type=Path,
        default=Path("samples"),
    )
    parser.add_argument("--jd", action="append", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/artifacts/features.csv"),
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    jd_texts = {
        path.stem: path.read_text(encoding="utf-8")
        for path in args.jd
    }
    rows = collect_feature_rows(
        args.labels,
        jd_texts,
        _resolve_resumes(args.resumes_dir),
        Settings(
            scoring_mode="v2",
            debug_endpoints=True,
            rate_limit_per_minute=100000,
        ),
    )
    jd_count = len({row["jd_id"] for row in rows})
    if len(rows) < 60 or jd_count < 5:
        raise SystemExit(
            "Insufficient labeled data: need at least 60 labeled pairs "
            f"and 5 JDs; found {len(rows)} pairs across {jd_count} JDs."
        )
    write_features(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main_cli()
