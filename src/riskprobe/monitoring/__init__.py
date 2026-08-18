"""Privacy-safe aggregate diagnostics and monitoring."""

from riskprobe.monitoring.drift import (
    diagnose_drift,
    population_stability_index,
    quantile_bin_edges,
)
from riskprobe.monitoring.models import (
    Alert,
    Diagnosis,
    DiagnosticReport,
    FeatureReference,
    FindingKind,
    FindingSeverity,
    ReferenceSnapshot,
    RiskFinding,
    RootCause,
    RuleReference,
    SafeProfile,
)
from riskprobe.monitoring.quality import diagnose_quality
from riskprobe.monitoring.reference import build_reference_snapshot
from riskprobe.monitoring.segments import diagnose_segments
from riskprobe.monitoring.service import diagnose_dataset
from riskprobe.monitoring.time import diagnose_time

__all__ = [
    "Alert",
    "Diagnosis",
    "DiagnosticReport",
    "FeatureReference",
    "FindingKind",
    "FindingSeverity",
    "ReferenceSnapshot",
    "RiskFinding",
    "RootCause",
    "RuleReference",
    "SafeProfile",
    "build_reference_snapshot",
    "diagnose_dataset",
    "diagnose_drift",
    "diagnose_quality",
    "diagnose_segments",
    "diagnose_time",
    "population_stability_index",
    "quantile_bin_edges",
]
