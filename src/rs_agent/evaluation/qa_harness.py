from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from rs_agent.tools.quality import MaskQualityGate


class QACase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    coverage_fraction: float = Field(ge=0, le=1)
    geometry_valid: bool = True
    input_valid: bool = True
    expected_flag: bool


class QAHarness:
    """Evaluates QA recall on fixed, labelled abnormal-result scenarios."""

    @staticmethod
    def run(case_dir: str | Path, minimum: float = 0.0, maximum: float = 0.98) -> dict:
        cases = [QACase.model_validate_json(path.read_text()) for path in sorted(Path(case_dir).glob("*.json"))]
        if not cases:
            raise ValueError("No QA evaluation cases found")
        rows = []
        for case in cases:
            assessment = MaskQualityGate.assess(
                case.coverage_fraction, minimum, maximum, case.geometry_valid, case.input_valid
            )
            rows.append({
                "case_id": case.case_id,
                "expected_flag": case.expected_flag,
                "flagged": not assessment.passed,
                "reasons": assessment.reasons,
            })
        positives = [row for row in rows if row["expected_flag"]]
        flagged = [row for row in rows if row["flagged"]]
        true_positive = [row for row in flagged if row["expected_flag"]]
        summary = {
            "cases": len(rows),
            "expected_anomalies": len(positives),
            "true_positives": len(true_positive),
            "false_negatives": len(positives) - len(true_positive),
            "false_positives": len(flagged) - len(true_positive),
            "recall": len(true_positive) / len(positives) if positives else None,
            "precision": len(true_positive) / len(flagged) if flagged else None,
            "flag_rate": mean(float(row["flagged"]) for row in rows),
        }
        return {"summary": summary, "cases": rows}

    @staticmethod
    def write(report: dict, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "qa_evaluation.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        csv_path = output / "qa_evaluation.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", "expected_flag", "flagged", "reasons"])
            writer.writeheader()
            writer.writerows(report["cases"])
        return {"json": json_path, "csv": csv_path}
