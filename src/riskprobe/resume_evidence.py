"""Generate internal resume drafts only from complete measured benchmark records."""

from dataclasses import dataclass

from riskprobe.benchmarking import BenchmarkRecord


@dataclass(frozen=True, slots=True)
class ResumeEvidence:
    task_count: int
    source_run_ids: tuple[str, ...]
    efficiency_rate: float
    total_reviewed_rules: int
    total_accepted_rules: int
    rule_review_acceptance: float | None
    anomaly_precision: float | None
    anomaly_recall: float | None
    root_cause_top3_hit_rate: float | None


@dataclass(frozen=True, slots=True)
class ResumeDraft:
    public_project: str
    company_experience: str


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_benchmarks(records: list[BenchmarkRecord]) -> ResumeEvidence:
    task_ids = {record.task_id for record in records}
    if len(task_ids) < 3:
        raise ValueError("at least 3 completed tasks with distinct task IDs are required")
    if len(task_ids) != len(records):
        raise ValueError("benchmark records must have unique task IDs")
    if any(record.manual_minutes is None for record in records):
        raise ValueError("every completed task requires a measured manual baseline")
    if any(record.baseline_fingerprint is None for record in records):
        raise ValueError("every benchmark record requires baseline provenance")
    if any(
        abs(record.agent_minutes - sum(item.seconds for item in record.stage_timings) / 60) > 1e-9
        for record in records
    ):
        raise ValueError("agent_minutes must equal recorded stage timings")
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("benchmark records must have unique run IDs")
    for field in ("dataset_id", "config_hash", "data_fingerprint"):
        if len({getattr(record, field) for record in records}) != 1:
            raise ValueError(f"benchmark records must share {field}")
    if any(
        record.baseline_task_id is not None and record.baseline_task_id != record.task_id
        for record in records
    ):
        raise ValueError("baseline task identity does not match benchmark task")
    agent_minutes = sum(record.agent_minutes for record in records)
    manual_minutes = sum(record.manual_minutes or 0.0 for record in records)
    if manual_minutes <= 0:
        raise ValueError("measured manual baselines must total more than zero")
    reviewed = sum(record.reviewed_rule_count for record in records)
    accepted = sum(record.accepted_rule_count for record in records)
    true_positive = sum(record.anomaly_true_positive_count for record in records)
    false_positive = sum(record.anomaly_false_positive_count for record in records)
    false_negative = sum(record.anomaly_false_negative_count for record in records)
    top3_hits = sum(record.root_cause_top3_hit_count for record in records)
    root_cause_cases = sum(record.root_cause_case_count for record in records)
    return ResumeEvidence(
        task_count=len(task_ids),
        source_run_ids=tuple(sorted(record.run_id for record in records)),
        efficiency_rate=(manual_minutes - agent_minutes) / manual_minutes,
        total_reviewed_rules=reviewed,
        total_accepted_rules=accepted,
        rule_review_acceptance=_ratio(accepted, reviewed),
        anomaly_precision=_ratio(true_positive, true_positive + false_positive),
        anomaly_recall=_ratio(true_positive, true_positive + false_negative),
        root_cause_top3_hit_rate=_ratio(top3_hits, root_cause_cases),
    )


def render_resume_bullets(evidence: ResumeEvidence) -> ResumeDraft:
    """Render a public project sentence and an internal, traceable company draft."""
    public_project = (
        "Built a local, auditable risk-rule workflow with evidence validation, "
        "privacy-preserving aggregate outputs, and human review gates."
    )
    sentences = [
        "Completed "
        f"{evidence.task_count} completed measured validation tasks, reducing "
        f"end-to-end workflow time by {evidence.efficiency_rate:.1%} against recorded manual baselines.",
        "Reviewed "
        f"{evidence.total_reviewed_rules} candidate rules and accepted "
        f"{evidence.total_accepted_rules} through documented human review.",
    ]
    if evidence.anomaly_recall is not None:
        sentences.append(f"Aggregated anomaly recall was {evidence.anomaly_recall:.1%}.")
    if evidence.root_cause_top3_hit_rate is not None:
        sentences.append(
            "Aggregated root-cause Top-3 hit rate was "
            f"{evidence.root_cause_top3_hit_rate:.1%}."
        )
    return ResumeDraft(public_project=public_project, company_experience=" ".join(sentences))


def render_markdown(evidence: ResumeEvidence) -> str:
    draft = render_resume_bullets(evidence)
    source_runs = "\n".join(f"- `{run_id}`" for run_id in evidence.source_run_ids)
    return (
        "# Measured resume evidence (internal)\n\n"
        "## Public project draft\n\n"
        f"{draft.public_project}\n\n"
        "## Company experience draft\n\n"
        f"{draft.company_experience}\n\n"
        "## Source run IDs\n\n"
        f"{source_runs}\n"
    )
