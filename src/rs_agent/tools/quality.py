from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    coverage_fraction: float
    reasons: list[str]


class MaskQualityGate:
    """Pure, auditable QA gate used by runtime QA and its labelled evaluation set."""

    @staticmethod
    def assess(
        coverage_fraction: float,
        minimum: float,
        maximum: float,
        geometry_valid: bool = True,
        input_valid: bool = True,
    ) -> QualityAssessment:
        reasons: list[str] = []
        if not input_valid:
            reasons.append("input_metadata_invalid")
        if not geometry_valid:
            reasons.append("output_geometry_invalid")
        if not minimum < coverage_fraction < maximum:
            reasons.append("empty_or_implausibly_dominant_mask")
        return QualityAssessment(not reasons, coverage_fraction, reasons)
