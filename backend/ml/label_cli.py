from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

STAGED_ORDER = [
    "samples_jd",
    "handwritten_00",
    "handwritten_01",
    "handwritten_02",
    "handwritten_04",
    "handwritten_03",
    "handwritten_07",
]

VALID_LABELS = {"Great", "Good", "Weak"}
KEY_MAP = {
    "g": "Great",
    "great": "Great",
    "m": "Good",
    "good": "Good",
    "w": "Weak",
    "weak": "Weak",
}


def parse_view(view_path: Path) -> tuple[str, dict[str, str]]:
    text = view_path.read_text(encoding="utf-8")
    jd_match = re.search(
        r"^## Job description\s*\n+(.*?)(?=\n## Resumes\b)",
        text,
        flags=re.S | re.M,
    )
    jd_text = jd_match.group(1).strip() if jd_match else ""

    blocks: dict[str, str] = {}
    matches = list(re.finditer(r"^### (\S+)\s*\n+", text, flags=re.M))
    for idx, match in enumerate(matches):
        rid = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks[rid] = text[start:end].strip()
    return jd_text, blocks


def atomic_save(labels_path: Path, rows: list[dict[str, str]]) -> None:
    tmp_path = labels_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["jd_id", "resume_file", "label"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, labels_path)


def print_summary(rows: list[dict[str, str]]) -> None:
    print("\n=== LABEL SUMMARY ===")
    total_counts: Counter[str] = Counter()
    jds_with_labels: set[str] = set()
    by_jd: dict[str, Counter[str]] = {}

    for row in rows:
        lbl = row["label"].strip() or "blank"
        by_jd.setdefault(row["jd_id"], Counter())[lbl] += 1
        total_counts[lbl] += 1
        if lbl in VALID_LABELS:
            jds_with_labels.add(row["jd_id"])

    ordered_jds = [j for j in STAGED_ORDER if j in by_jd] + [
        j for j in by_jd if j not in STAGED_ORDER
    ]
    for jd in ordered_jds:
        c = by_jd[jd]
        print(
            f"  {jd:<16} Great={c['Great']:2d}  Good={c['Good']:2d}  "
            f"Weak={c['Weak']:2d}  blank={c['blank']:2d}"
        )

    labeled = sum(total_counts[k] for k in VALID_LABELS)
    pos = total_counts["Great"] + total_counts["Good"]
    weak = total_counts["Weak"]
    print(
        f"\nTotal labeled: {labeled}/{len(rows)} across {len(jds_with_labels)} JDs "
        f"(Great+Good={pos}, Weak={weak}, blank={total_counts['blank']})"
    )
    if labeled >= 60 and len(jds_with_labels) >= 5 and pos >= 15 and weak >= 15:
        print("Status: READY for fit_scoring.py (clears 60-pair / 5-JD / balance minimum).")
    else:
        print("Status: NOT YET READY (need >=60 pairs across >=5 JDs, >=15 Great+Good, >=15 Weak).")


def main() -> None:
    artifacts = Path("ml/artifacts")
    template_path = artifacts / "labels_template.csv"
    labels_path = artifacts / "labels.csv"
    view_dir = artifacts / "labeling_view"

    if not labels_path.exists():
        if not template_path.exists():
            sys.exit(f"Missing template: {template_path}")
        labels_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Initialized {labels_path} from {template_path}")

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    target_jd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    include_labeled = "--all" in sys.argv

    order_rank = {jd: i for i, jd in enumerate(STAGED_ORDER)}
    queue = [
        idx
        for idx, row in enumerate(rows)
        if (target_jd is None or row["jd_id"] == target_jd)
        and (include_labeled or row["label"].strip() not in VALID_LABELS)
    ]
    queue.sort(key=lambda idx: (order_rank.get(rows[idx]["jd_id"], 999), idx))

    if not queue:
        print("Nothing left to label for the selected filter.")
        print_summary(rows)
        return

    views_cache: dict[str, tuple[str, dict[str, str]]] = {}
    current_jd = None
    pos = 0

    while 0 <= pos < len(queue):
        row_idx = queue[pos]
        row = rows[row_idx]
        jd_id = row["jd_id"]
        rid = row["resume_file"]

        if jd_id not in views_cache:
            views_cache[jd_id] = parse_view(view_dir / f"{jd_id}.md")
        jd_text, blocks = views_cache[jd_id]

        if jd_id != current_jd:
            current_jd = jd_id
            print("\n" + "=" * 78)
            print(f"JOB DESCRIPTION: {jd_id}")
            print("=" * 78)
            print(jd_text)
            print("=" * 78)

        resume_text = blocks.get(rid, "[MISSING BLOCK IN VIEW]")
        existing = row["label"].strip() or "blank"
        print(f"\n[{pos + 1}/{len(queue)}] JD: {jd_id} | ID: {rid} | Current: {existing}")
        print("-" * 78)
        print(resume_text)
        print("-" * 78)

        while True:
            raw = input("Label [g=Great, m=Good, w=Weak, s=skip, b=back, j=JD, c=counts, q=quit]: ").strip().lower()
            if raw in KEY_MAP:
                row["label"] = KEY_MAP[raw]
                try:
                    atomic_save(labels_path, rows)
                except PermissionError:
                    print("ERROR: Could not write labels.csv (close it if open in Excel) and retry.")
                    continue
                pos += 1
                break
            if raw == "s":
                pos += 1
                break
            if raw == "b":
                if pos > 0:
                    pos -= 1
                else:
                    print("Already at the first item in this session.")
                break
            if raw == "j":
                print("\n" + "=" * 78)
                print(f"JOB DESCRIPTION: {jd_id}")
                print("=" * 78)
                print(jd_text)
                print("=" * 78)
            elif raw == "c":
                print_summary(rows)
            elif raw == "q":
                print_summary(rows)
                return
            else:
                print("Invalid key. Use g, m, w, s, b, j, c, or q.")

    print("\nAll queued items finished!")
    print_summary(rows)


if __name__ == "__main__":
    main()