from __future__ import annotations

import csv
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
OUT = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "mmlu.jsonl"
TMP = OUT.parent / "mmlu_data.tar"

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def download_archive() -> None:
    print(f"Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "forge-build/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        TMP.write_bytes(response.read())
    print(f"  -> {TMP.name} ({TMP.stat().st_size} bytes)")


def normalize_row(subject: str, row: list[str], index: int) -> dict:
    return {
        "id": f"{subject}:{index}",
        "subject": subject,
        "question": row[0],
        "labels": ["A", "B", "C", "D"],
        "choices": row[1:5],
        "answer": row[5].strip().upper(),
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    download_archive()

    rows_written = 0
    with tarfile.open(TMP) as archive, OUT.open("w", encoding="utf-8") as out:
        for subject in SUBJECTS:
            member = archive.getmember(f"data/test/{subject}_test.csv")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)

            lines = (line.decode("utf-8") for line in extracted)
            reader = csv.reader(lines)
            for index, row in enumerate(reader):
                out.write(json.dumps(normalize_row(subject, row, index), ensure_ascii=False) + "\n")
                rows_written += 1

    TMP.unlink(missing_ok=True)
    print(f"Wrote {rows_written} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
