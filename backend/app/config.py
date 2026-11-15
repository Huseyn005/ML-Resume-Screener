from functools import lru_cache
from math import isclose, isfinite
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.model_files import model_folder_issues

BACKEND_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_DIR = BACKEND_ROOT / "models" / "finetuned"
CALIBRATOR_PATH = FINETUNED_DIR / "calibrator.pkl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allowed_origins: str = (
        "https://pushcv.vercel.app, https://statcv.vercel.app, "
        "https://pullcv.vercel.app, https://taphire.vercel.app, "
        "https://dragcv.vercel.app,https://shortlists.vercel.app,"
        "http://localhost:5173,http://localhost:8080"
    )
    allowed_origin_regex: str = ""
    log_level: str = "INFO"
    finetuned_repo: str = ""
    finetuned_revision: str = ""
    hf_token: str = ""
    semantic_weight: float = 0.5
    skill_weight: float = 0.40
    section_weight: float = 0.10
    experience_skill_weight: float = 1.0
    project_skill_weight: float = 1.0
    skills_skill_weight: float = 0.6
    summary_skill_weight: float = 0.3
    education_skill_weight: float = 0.3
    other_skill_weight: float = 0.3
    debug_endpoints: bool = False
    scoring_mode: str = "legacy"
    semantic_source: str = "calibrated"
    scoring_display_map: str = ""
    semantic_low: float = 0.20
    semantic_high: float = 0.75
    gate_floor: float = 0.5
    gate_gamma: float = 1.0
    semantic_skill_credit: float = 0.6
    unlisted_requirement_share: float = 0.2
    v2_phrase_cap: int = 1024
    v2_weight_reliability_skills: float | None = None
    v2_gate_reliability_skills: float | None = None
    section_low: float = 0.0
    section_high: float = 1.0
    requirement_low: float = 0.0
    requirement_high: float = 1.0
    skill_cosine_threshold: float = 0.72
    min_hard_skills: int = 4
    chunk_tokens: int = 200
    chunk_overlap: int = 40
    max_pdf_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 20
    max_request_bytes: int = 60 * 1024 * 1024
    max_files: int = 30
    max_concurrent_analyses: int = 2
    queue_timeout_seconds: float = 30.0
    rate_limit_per_minute: int = 20
    trusted_proxy_count: int = 1
    min_text_chars: int = 200
    hidden_font_size: float = 2.0

    @model_validator(mode="after")
    def validate_score_weights(self) -> "Settings":
        if self.v2_weight_reliability_skills is None:
            object.__setattr__(
                self,
                "v2_weight_reliability_skills",
                float(self.min_hard_skills),
            )
        if self.v2_gate_reliability_skills is None:
            object.__setattr__(
                self,
                "v2_gate_reliability_skills",
                float(self.min_hard_skills),
            )
        if self.scoring_mode not in {"legacy", "v2"}:
            raise ValueError("scoring_mode must be legacy or v2")
        if self.semantic_source not in {"calibrated", "window"}:
            raise ValueError(
                "semantic_source must be calibrated or window"
            )
        if self.scoring_display_map.strip():
            try:
                pairs = []
                for item in self.scoring_display_map.split(","):
                    parts = item.strip().split(":")
                    if len(parts) != 2:
                        raise ValueError
                    x, y = (float(value.strip()) for value in parts)
                    if not isfinite(x) or not isfinite(y):
                        raise ValueError
                    pairs.append((x, y))
            except (TypeError, ValueError):
                raise ValueError(
                    "scoring_display_map must contain finite x:y pairs"
                ) from None
            if (
                len(pairs) < 2
                or pairs[0][0] != 0.0
                or pairs[-1][0] != 1.0
                or any(left[0] >= right[0]
                       for left, right in zip(pairs, pairs[1:]))
                or any(
                    y < 0.0 or y > 100.0
                    for _, y in pairs
                )
                or any(
                    left[1] > right[1]
                    for left, right in zip(pairs, pairs[1:])
                )
            ):
                raise ValueError(
                    "scoring_display_map must start at x=0, end at x=1, "
                    "have increasing x values, and non-decreasing y values "
                    "in [0, 100]"
                )
        if not 0.0 <= self.gate_floor <= 1.0:
            raise ValueError("gate_floor must be between 0 and 1")
        if self.gate_gamma <= 0.0:
            raise ValueError("gate_gamma must be positive")
        if not 0.0 <= self.semantic_skill_credit <= 1.0:
            raise ValueError(
                "semantic_skill_credit must be between 0 and 1"
            )
        if not 0.0 <= self.unlisted_requirement_share <= 0.5:
            raise ValueError(
                "unlisted_requirement_share must be between 0 and 0.5"
            )
        if self.v2_phrase_cap < 1:
            raise ValueError("v2_phrase_cap must be at least 1")
        if self.v2_weight_reliability_skills <= 0.0:
            raise ValueError(
                "v2_weight_reliability_skills must be positive"
            )
        if self.v2_gate_reliability_skills <= 0.0:
            raise ValueError(
                "v2_gate_reliability_skills must be positive"
            )
        for name, low, high in (
            ("section", self.section_low, self.section_high),
            ("requirement", self.requirement_low, self.requirement_high),
        ):
            if low >= high:
                raise ValueError(f"{name}_low must be less than {name}_high")
        if self.min_hard_skills < 1:
            raise ValueError("min_hard_skills must be at least 1")
        if min(
            self.max_pdf_bytes,
            self.max_pdf_pages,
            self.max_request_bytes,
            self.max_files,
            self.max_concurrent_analyses,
        ) < 1:
            raise ValueError("analysis limits must be positive")
        if self.queue_timeout_seconds < 0:
            raise ValueError("queue_timeout_seconds cannot be negative")
        if self.rate_limit_per_minute < 0:
            raise ValueError("rate_limit_per_minute cannot be negative")
        if self.trusted_proxy_count < 0:
            raise ValueError("trusted_proxy_count cannot be negative")
        if self.log_level.upper() not in {
            "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG",
        }:
            raise ValueError("log_level must be a valid Python log level")
        total = (
            self.semantic_weight
            + self.skill_weight
            + self.section_weight
        )
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "semantic_weight, skill_weight, and section_weight "
                f"must sum to 1.0 (got {total:.6g})"
            )
        evidence_weights = (
            self.experience_skill_weight,
            self.project_skill_weight,
            self.skills_skill_weight,
            self.summary_skill_weight,
            self.education_skill_weight,
            self.other_skill_weight,
        )
        if any(not 0.0 <= weight <= 1.0 for weight in evidence_weights):
            raise ValueError("skill evidence weights must be between 0 and 1")
        return self

    def origin_list(self) -> list[str]:
        return [p.strip() for p in self.allowed_origins.split(",")
                if p.strip()]

    def model_path(self) -> str:
        if FINETUNED_DIR.is_dir() and not model_folder_issues(
            FINETUNED_DIR, CALIBRATOR_PATH.name
        ):
            return str(FINETUNED_DIR)
        return self.embed_model

    def calibrator_path(self) -> Path:
        return CALIBRATOR_PATH

    def model_source(self) -> str:
        if self.finetuned_repo:
            revision = self.finetuned_revision or "latest"
            return f"{self.finetuned_repo}@{revision}"
        if self.model_path() == str(FINETUNED_DIR):
            return "local-finetuned"
        return "base"


@lru_cache
def get_settings() -> Settings:
    return Settings()
