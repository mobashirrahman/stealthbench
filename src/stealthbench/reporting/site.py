"""Static model pages, comparisons and history (G13 T13A).

Contracts (``IMPLEMENTATION_PLAN.md`` G13, ``docs/contracts.md``):

* Observation dates, sample sizes, confidence intervals, costs, missing
  coverage and identity resolution are shown together. A missing measurement
  stays ``null``/empty, never ``0``; an incomplete core has no headline index.
* Predicted and revealed identities are separate sections. Similarity is never
  displayed as probability; unknown stays ``unknown``/``insufficient_evidence``.
* Every displayed score links to reproducible provenance (manifest hash, grade
  digest, plan digest, artifact counts). Unsupported claims are absent.
* All dynamic content is redacted then HTML-escaped, so script-bearing aliases
  or responses cannot execute. Artifact links cannot traverse outside the
  report directory.
* Review serving binds loopback only. Deployment is a separate authorized
  action and is not performed here.

Offline by construction: file reads plus atomic writes. No transport,
no credential, no subprocess. The only socket is the loopback review server.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from stealthbench.analysis.statistics import mean_ci, paired_difference_ci
from stealthbench.reporting.exports import (
    REPORT_CSV_NAME,
    REPORT_JSON_NAME,
    safe_output_path,
    sanitize_csv_cell,
    write_text_atomic,
)
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.manifest import manifest_hash
from stealthbench.schemas.results import IdentityReport
from stealthbench.storage.events import EventLog, redact_text

REPORT_VERSION: Final[str] = "1.0"
STALE_AFTER_DAYS: Final[int] = 30
CI_RE_SAMPLES: Final[int] = 1000
CI_CONFIDENCE: Final[float] = 0.95

LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "::1", "localhost"})

_SLUG_BAD: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


def safe_slug(value: str) -> str:
    """A filesystem-safe basename for an endpoint id.

    Anything outside ``[A-Za-z0-9._-]`` becomes ``_`` so a hostile id such as
    ``../../etc/passwd`` cannot escape the report directory. Empty results
    become ``endpoint``.
    """
    cleaned = _SLUG_BAD.sub("_", value).strip("._")
    if not cleaned:
        return "endpoint"
    return cleaned[:64]


def esc(value: object) -> str:
    """HTML-escape a value for element content or a quoted attribute."""
    return html.escape(str(value), quote=True)


def redacted_esc(value: object, *, extra_secrets: frozenset[str]) -> str:
    """Redact secrets first, then HTML-escape.

    Redaction must precede escaping so a secret containing ``<`` cannot
    survive inside an escaped entity.
    """
    return html.escape(redact_text(str(value), extra_secrets=extra_secrets), quote=True)


def _resolve_inside(output_dir: Path, name: str) -> Path:
    """Confine a report filename to ``output_dir`` (traversal guard)."""
    return safe_output_path(output_dir, name)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    """One endpoint's score on one benchmark."""

    benchmark_id: str
    category: str
    n_total: int
    n_graded: int
    n_correct: int
    n_missing: int
    accuracy: float | None
    ci_low: float | None
    ci_high: float | None
    ci_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "benchmark_id": self.benchmark_id,
            "category": self.category,
            "ci_available": self.ci_available,
            "ci_high": self.ci_high,
            "ci_low": self.ci_low,
            "n_correct": self.n_correct,
            "n_graded": self.n_graded,
            "n_missing": self.n_missing,
            "n_total": self.n_total,
        }


@dataclass(frozen=True, slots=True)
class EndpointSummary:
    """Everything one model page shows for one endpoint."""

    endpoint_id: str
    alias: str
    route: str
    slug: str
    n_total: int
    n_graded: int
    n_correct: int
    n_missing: int
    n_transport_failed: int
    accuracy: float | None
    ci_low: float | None
    ci_high: float | None
    ci_available: bool
    coverage: str
    input_tokens: int | None
    output_tokens: int | None
    cost_billed: str | None
    cost_spent: str | None
    identity_state: str
    abstention_reason: str | None
    predicted: tuple[dict[str, Any], ...]
    reveal_label: str | None
    stale: bool
    observation_started: str | None
    observation_ended: str | None
    provenance_anchor: str
    benchmarks: tuple[BenchmarkCell, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstention_reason": self.abstention_reason,
            "accuracy": self.accuracy,
            "alias": self.alias,
            "benchmarks": [cell.to_dict() for cell in self.benchmarks],
            "ci_available": self.ci_available,
            "ci_high": self.ci_high,
            "ci_low": self.ci_low,
            "cost_billed": self.cost_billed,
            "cost_spent": self.cost_spent,
            "coverage": self.coverage,
            "endpoint_id": self.endpoint_id,
            "identity_state": self.identity_state,
            "input_tokens": self.input_tokens,
            "n_correct": self.n_correct,
            "n_graded": self.n_graded,
            "n_missing": self.n_missing,
            "n_total": self.n_total,
            "n_transport_failed": self.n_transport_failed,
            "observation_ended": self.observation_ended,
            "observation_started": self.observation_started,
            "output_tokens": self.output_tokens,
            "predicted": [dict(item) for item in self.predicted],
            "provenance_anchor": self.provenance_anchor,
            "reveal_label": self.reveal_label,
            "route": self.route,
            "slug": self.slug,
            "stale": self.stale,
        }


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """One paired comparison on shared declared items."""

    first_id: str
    second_id: str
    benchmark_id: str
    n_shared_tasks: int
    observed_difference: float | None
    ci_low: float | None
    ci_high: float | None
    available: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "benchmark_id": self.benchmark_id,
            "ci_high": self.ci_high,
            "ci_low": self.ci_low,
            "first_id": self.first_id,
            "n_shared_tasks": self.n_shared_tasks,
            "observed_difference": self.observed_difference,
            "second_id": self.second_id,
            "verdict": self.verdict,
        }


@dataclass(slots=True)
class ReportData:
    """The full in-memory report before rendering."""

    campaign_id: str
    manifest_hash: str
    score_version: str
    seed: int
    grade_digest: str | None
    plan_digest: str | None
    event_count: int
    accepted_samples: int
    observation_started: str | None
    observation_ended: str | None
    endpoints: list[EndpointSummary] = field(default_factory=list)
    comparisons: list[PairwiseComparison] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    stale_campaign: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_samples": self.accepted_samples,
            "campaign_id": self.campaign_id,
            "comparisons": [item.to_dict() for item in self.comparisons],
            "endpoints": [item.to_dict() for item in self.endpoints],
            "event_count": self.event_count,
            "grade_digest": self.grade_digest,
            "ledger": dict(self.ledger),
            "manifest_hash": self.manifest_hash,
            "observation_ended": self.observation_ended,
            "observation_started": self.observation_started,
            "plan_digest": self.plan_digest,
            "provenance": dict(self.provenance),
            "report_version": REPORT_VERSION,
            "score_version": self.score_version,
            "seed": self.seed,
            "stale_campaign": self.stale_campaign,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _is_stale(ended_at: str | None, *, now: datetime) -> bool:
    if ended_at is None:
        return False
    try:
        moment = datetime.fromisoformat(ended_at)
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    delta = now - moment.astimezone(UTC)
    return delta.days > STALE_AFTER_DAYS


def _load_identities(
    artifacts_dir: Path,
) -> tuple[dict[str, IdentityReport], dict[str, dict[str, Any]]]:
    """Load optional ``identity/<endpoint_id>.json`` reports.

    Missing or unparsable files mean ``unknown``; they never raise here
    because an absent identity signal is a display state, not a build error.
    """
    reports: dict[str, IdentityReport] = {}
    raw: dict[str, dict[str, Any]] = {}
    identity_dir = artifacts_dir / "identity"
    if not identity_dir.is_dir():
        return reports, raw
    for child in sorted(identity_dir.iterdir()):
        if not child.is_file() or child.suffix != ".json":
            continue
        try:
            payload = _read_json(child)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            report = IdentityReport.model_validate(payload)
        except ValueError:
            continue
        reports[report.endpoint_id] = report
        raw[report.endpoint_id] = payload
    return reports, raw


def build_report_data(
    artifacts_dir: Path, *, extra_secrets: frozenset[str] = frozenset()
) -> ReportData:
    """Assemble scores, intervals, costs, coverage and identity for rendering.

    Reads only local files. Raises ``ValueError`` when ``artifacts_dir`` is
    not a campaign artifact directory.
    """
    root = Path(artifacts_dir)
    if not root.is_dir():
        raise ValueError(f"{artifacts_dir} is not an artifact directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{artifacts_dir} is not an artifact directory: manifest.json missing")
    manifest = CampaignManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    digest = manifest_hash(manifest)

    summary: dict[str, Any] = {}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        loaded = _read_json(summary_path)
        if isinstance(loaded, dict):
            summary = loaded
    grade_digest = summary.get("grade_digest")
    if not isinstance(grade_digest, str):
        grade_digest = None
    plan_digest = summary.get("plan_digest")
    if not isinstance(plan_digest, str):
        plan_digest = None

    log = EventLog(root, extra_secrets=extra_secrets)
    events, integrity = log.read_all()

    grade_rows = _read_jsonl(root / "grades.jsonl")
    generation_rows = _read_jsonl(root / "export.jsonl")

    ledger: dict[str, Any] = {}
    ledger_path = root / "ledger.json"
    if ledger_path.is_file():
        loaded_ledger = _read_json(ledger_path)
        if isinstance(loaded_ledger, dict):
            ledger = loaded_ledger

    identities, _raw_identities = _load_identities(root)

    window = manifest.observation_window
    started = window.started_at.isoformat() if window.started_at is not None else None
    ended = window.ended_at.isoformat() if window.ended_at is not None else None
    now = datetime.now(UTC)
    stale_campaign = _is_stale(ended, now=now)

    benchmark_to_category = {spec.benchmark_id: spec.category for spec in manifest.benchmarks}
    benchmark_planned: dict[str, dict[str, int]] = {}
    for spec in manifest.benchmarks:
        planned_items = len(spec.item_ids) if spec.item_ids else 0
        benchmark_planned[spec.benchmark_id] = {
            "items": planned_items,
            "repeats": spec.repeats,
        }

    # Group grade rows by endpoint -> benchmark -> task -> outcomes.
    # strict_correctness pass/fail -> 1.0/0.0; anything else excluded.
    per_endpoint_benchmark_task: dict[str, dict[str, dict[str, list[float]]]] = {}
    per_endpoint_counts: dict[str, dict[str, int]] = {}
    for row in grade_rows:
        key = row.get("sample_key")
        if not isinstance(key, dict):
            continue
        endpoint_id = str(key.get("endpoint_id", ""))
        task_id = str(key.get("task_id", ""))
        if not endpoint_id or not task_id:
            continue
        benchmark_id = task_id.split("::", 1)[0] if "::" in task_id else task_id
        transport = str(row.get("transport", ""))
        evaluator = str(row.get("evaluator", ""))
        strict = str(row.get("strict_correctness", ""))
        counts = per_endpoint_counts.setdefault(
            endpoint_id, {"graded": 0, "correct": 0, "transport_failed": 0}
        )
        if transport != "pass":
            counts["transport_failed"] += 1
            continue
        if evaluator != "pass":
            continue
        if strict not in {"pass", "fail"}:
            continue
        counts["graded"] += 1
        if strict == "pass":
            counts["correct"] += 1
        outcome = 1.0 if strict == "pass" else 0.0
        per_endpoint_benchmark_task.setdefault(endpoint_id, {}).setdefault(
            benchmark_id, {}
        ).setdefault(task_id, []).append(outcome)

    # Per-endpoint token sums from stored generations.
    per_endpoint_tokens: dict[str, dict[str, Any]] = {}
    for row in generation_rows:
        key = row.get("sample_key")
        if not isinstance(key, dict):
            continue
        endpoint_id = str(key.get("endpoint_id", ""))
        if not endpoint_id:
            continue
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        entry = per_endpoint_tokens.setdefault(
            endpoint_id,
            {"input": 0, "output": 0, "has_input": False, "has_output": False},
        )
        in_tokens = usage.get("input_tokens")
        out_tokens = usage.get("output_tokens")
        if isinstance(in_tokens, int) and not isinstance(in_tokens, bool):
            current_in = entry["input"]
            base_in = current_in if isinstance(current_in, int) else 0
            entry["input"] = base_in + in_tokens
            entry["has_input"] = True
        if isinstance(out_tokens, int) and not isinstance(out_tokens, bool):
            current_out = entry["output"]
            base_out = current_out if isinstance(current_out, int) else 0
            entry["output"] = base_out + out_tokens
            entry["has_output"] = True

    endpoints: list[EndpointSummary] = []
    for endpoint in manifest.endpoints:
        endpoint_id = endpoint.endpoint_id
        counts = per_endpoint_counts.get(endpoint_id, {"graded": 0, "correct": 0})
        graded = int(counts.get("graded", 0))
        correct = int(counts.get("correct", 0))
        transport_failed = int(counts.get("transport_failed", 0))
        # Planned total across all declared benchmarks for this endpoint.
        planned_total = sum(
            benchmark_planned[spec.benchmark_id]["items"]
            * benchmark_planned[spec.benchmark_id]["repeats"]
            for spec in manifest.benchmarks
        )
        missing = max(0, planned_total - graded - transport_failed)
        # Overall accuracy over graded samples (response-conditional).
        accuracy = (correct / graded) if graded else None

        # Overall CI over task-clustered outcomes across benchmarks.
        all_tasks: dict[str, list[float]] = {}
        for benchmark_tasks in per_endpoint_benchmark_task.get(endpoint_id, {}).values():
            for task_id, outcomes in benchmark_tasks.items():
                all_tasks.setdefault(task_id, []).extend(outcomes)
        # Collapse repeats to per-task means for the CI input.
        ci_input: dict[str, list[float]] = {}
        for task_id, outcomes in all_tasks.items():
            if outcomes:
                ci_input[task_id] = [sum(outcomes) / len(outcomes)]
        interval = mean_ci(
            ci_input,
            seed=manifest.seed,
            n_resamples=CI_RE_SAMPLES,
            confidence=CI_CONFIDENCE,
        )
        if interval.available:
            ci_low, ci_high = interval.ci_low, interval.ci_high
        else:
            ci_low, ci_high = None, None

        tokens = per_endpoint_tokens.get(endpoint_id, {})
        has_input = bool(tokens.get("has_input", False))
        has_output = bool(tokens.get("has_output", False))
        raw_in = tokens.get("input", 0)
        raw_out = tokens.get("output", 0)
        input_tokens = raw_in if (has_input and isinstance(raw_in, int)) else None
        output_tokens = raw_out if (has_output and isinstance(raw_out, int)) else None

        identity = identities.get(endpoint_id)
        if identity is None:
            identity_state = "unknown"
            abstention: str | None = "insufficient_evidence"
            predicted: tuple[dict[str, Any], ...] = ()
            reveal: str | None = None
        else:
            if identity.ranked_similarities:
                identity_state = "similarity_ranked"
                abstention = identity.abstention_reason
            else:
                identity_state = "unknown"
                abstention = identity.abstention_reason or "insufficient_evidence"
            predicted = tuple(
                {
                    "candidate_id": entry.candidate_id,
                    "similarity": float(entry.similarity),
                    "evidence_signals": list(entry.evidence_signals),
                    "note": entry.note,
                }
                for entry in identity.ranked_similarities
            )
            reveal = identity.official_reveal_label

        if graded == 0 and planned_total > 0:
            coverage = "failed" if transport_failed else "missing"
        elif graded < planned_total:
            coverage = "partial"
        else:
            coverage = "complete"

        slug = safe_slug(endpoint_id)
        anchor = f"endpoint-{slug}"
        cells: list[BenchmarkCell] = []
        for spec in manifest.benchmarks:
            task_map = per_endpoint_benchmark_task.get(endpoint_id, {}).get(spec.benchmark_id, {})
            b_graded_outcomes: list[float] = []
            b_correct = 0
            b_graded = 0
            for outcomes in task_map.values():
                for outcome in outcomes:
                    b_graded += 1
                    b_graded_outcomes.append(outcome)
                    if outcome == 1.0:
                        b_correct += 1
            planned = benchmark_planned[spec.benchmark_id]["items"] * spec.repeats
            # Transport failures for this benchmark are not in task_map; derive
            # from the endpoint total proportionally? Keep simple: missing =
            # planned - graded samples for this benchmark (failures included).
            b_missing = max(0, planned - b_graded)
            b_accuracy = (b_correct / b_graded) if b_graded else None
            b_ci_input = {task: [sum(v) / len(v)] for task, v in task_map.items() if v}
            b_interval = mean_ci(
                b_ci_input,
                seed=manifest.seed,
                n_resamples=CI_RE_SAMPLES,
                confidence=CI_CONFIDENCE,
            )
            if b_interval.available:
                b_low, b_high = b_interval.ci_low, b_interval.ci_high
                b_avail = True
            else:
                b_low, b_high, b_avail = None, None, False
            cells.append(
                BenchmarkCell(
                    benchmark_id=spec.benchmark_id,
                    category=spec.category,
                    n_total=planned,
                    n_graded=b_graded,
                    n_correct=b_correct,
                    n_missing=b_missing,
                    accuracy=b_accuracy,
                    ci_low=b_low,
                    ci_high=b_high,
                    ci_available=b_avail,
                )
            )
        billed: str | None = None
        spent: str | None = None
        if isinstance(ledger, dict):
            raw_billed = ledger.get("billed")
            raw_spent = ledger.get("spent")
            billed = str(raw_billed) if raw_billed is not None else None
            spent = str(raw_spent) if raw_spent is not None else None
        # Category label for the benchmark map (unused beyond display).
        _ = benchmark_to_category
        endpoints.append(
            EndpointSummary(
                endpoint_id=endpoint_id,
                alias=endpoint.alias,
                route=endpoint.route,
                slug=slug,
                n_total=planned_total,
                n_graded=graded,
                n_correct=correct,
                n_missing=missing,
                n_transport_failed=transport_failed,
                accuracy=accuracy,
                ci_low=ci_low,
                ci_high=ci_high,
                ci_available=bool(interval.available),
                coverage=coverage,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_billed=billed,
                cost_spent=spent,
                identity_state=identity_state,
                abstention_reason=abstention,
                predicted=predicted,
                reveal_label=reveal,
                stale=stale_campaign,
                observation_started=started,
                observation_ended=ended,
                provenance_anchor=anchor,
                benchmarks=tuple(cells),
            )
        )

    endpoints.sort(
        key=lambda item: (
            item.accuracy is None,
            -(item.accuracy if item.accuracy is not None else 0.0),
            item.endpoint_id,
        )
    )

    # Pairwise comparisons on shared tasks (strict outcomes, first benchmark
    # with shared tasks, else overall).
    comparisons: list[PairwiseComparison] = []
    endpoint_ids = [item.endpoint_id for item in endpoints]
    for index, first_id in enumerate(endpoint_ids):
        for second_id in endpoint_ids[index + 1 :]:
            first_tasks = per_endpoint_benchmark_task.get(first_id, {})
            second_tasks = per_endpoint_benchmark_task.get(second_id, {})
            shared_benchmarks = sorted(set(first_tasks) & set(second_tasks))
            benchmark_choice = shared_benchmarks[0] if shared_benchmarks else None
            if benchmark_choice is not None:
                first_map = first_tasks[benchmark_choice]
                second_map = second_tasks[benchmark_choice]
                label = benchmark_choice
            else:
                # Fall back to overall task maps.
                first_map = {}
                for task_map in first_tasks.values():
                    first_map.update(task_map)
                second_map = {}
                for task_map in second_tasks.values():
                    second_map.update(task_map)
                label = "overall"
            # mean_ci inputs are per-task single means; paired_difference_ci
            # expects task -> repeats, so wrap each mean as one repeat.
            first_wrapped = {task: list(values) for task, values in first_map.items()}
            second_wrapped = {task: list(values) for task, values in second_map.items()}
            paired = paired_difference_ci(
                first_wrapped,
                second_wrapped,
                seed=manifest.seed,
                n_resamples=CI_RE_SAMPLES,
                confidence=CI_CONFIDENCE,
            )
            if paired.available:
                low, high = paired.ci_low, paired.ci_high
                observed = paired.observed_difference
                assert low is not None and high is not None and observed is not None
                if low > 0:
                    verdict = "first_leads"
                elif high < 0:
                    verdict = "second_leads"
                else:
                    verdict = "tied_or_uncertain"
            else:
                low, high, observed = None, None, None
                verdict = "unavailable"
            comparisons.append(
                PairwiseComparison(
                    first_id=first_id,
                    second_id=second_id,
                    benchmark_id=label,
                    n_shared_tasks=paired.n_shared_tasks,
                    observed_difference=observed,
                    ci_low=low,
                    ci_high=high,
                    available=paired.available,
                    verdict=verdict,
                )
            )

    accepted = sum(1 for row in grade_rows if str(row.get("transport", "")) == "pass")
    provenance: dict[str, Any] = {
        "artifacts_dir": str(root),
        "event_count": len(events),
        "grade_digest": grade_digest,
        "integrity": integrity.to_dict(),
        "manifest_hash": digest,
        "plan_digest": plan_digest,
        "report_version": REPORT_VERSION,
        "score_version": manifest.score_version,
    }
    return ReportData(
        campaign_id=manifest.campaign_id,
        manifest_hash=digest,
        score_version=manifest.score_version,
        seed=manifest.seed,
        grade_digest=grade_digest,
        plan_digest=plan_digest,
        event_count=len(events),
        accepted_samples=accepted,
        observation_started=started,
        observation_ended=ended,
        endpoints=endpoints,
        comparisons=comparisons,
        ledger=dict(ledger),
        provenance=provenance,
        stale_campaign=stale_campaign,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _score_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _score_link(text: str, anchor: str) -> str:
    return f'<a href="provenance.html#{esc(anchor)}">{esc(text)}</a>'


def _page_shell(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{esc(title)}</title>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{esc(title)}</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def render_index(data: ReportData, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Leaderboard with every score linking to provenance."""
    rows: list[str] = []
    for rank, endpoint in enumerate(data.endpoints, start=1):
        score = _score_text(endpoint.accuracy)
        linked = _score_link(score, endpoint.provenance_anchor)
        ci = (
            f"{endpoint.ci_low:.3f}-{endpoint.ci_high:.3f}"
            if endpoint.ci_available and endpoint.ci_low is not None
            else "n/a"
        )
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f'<td><a href="endpoint-{esc(endpoint.slug)}.html">'
            f"{redacted_esc(endpoint.alias, extra_secrets=extra_secrets)}</a></td>"
            f"<td>{linked}</td>"
            f"<td>{esc(ci)}</td>"
            f"<td>{endpoint.n_graded}/{endpoint.n_total}</td>"
            f"<td>{esc(endpoint.coverage)}</td>"
            f"<td>{esc(endpoint.identity_state)}</td>"
            "</tr>"
        )
    stale_note = ""
    if data.stale_campaign:
        stale_note = "<p><strong>Stale evidence:</strong> observation window ended "
        stale_note += f"more than {STALE_AFTER_DAYS} days ago.</p>"
    body = (
        f"<p>Campaign {redacted_esc(data.campaign_id, extra_secrets=extra_secrets)} "
        f"manifest {esc(data.manifest_hash)} "
        f"score version {esc(data.score_version)}.</p>"
        f"<p>Observation window: {esc(data.observation_started)} "
        f"to {esc(data.observation_ended)}. "
        f"Samples: {data.accepted_samples} accepted over {data.event_count} events.</p>"
        f"{stale_note}"
        "<h2>Leaderboard</h2>"
        "<table><thead><tr><th>rank</th><th>model</th><th>accuracy</th>"
        "<th>95% CI</th><th>samples</th><th>coverage</th><th>identity</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<p><a href="comparison.html">Comparisons</a> | '
        '<a href="history.html">History</a> | '
        '<a href="provenance.html">Provenance</a></p>'
    )
    return _page_shell(f"StealthBench report {data.campaign_id}", body)


def render_endpoint(
    data: ReportData,
    endpoint: EndpointSummary,
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> str:
    """One static model page with dates, sizes, CIs, costs, coverage, identity."""
    score = _score_text(endpoint.accuracy)
    linked = _score_link(score, endpoint.provenance_anchor)
    ci = (
        f"{endpoint.ci_low:.3f}-{endpoint.ci_high:.3f}"
        if endpoint.ci_available and endpoint.ci_low is not None
        else "n/a (insufficient tasks for a task-level interval)"
    )
    benchmark_rows: list[str] = []
    for cell in endpoint.benchmarks:
        cell_score = _score_link(
            _score_text(cell.accuracy), f"{endpoint.provenance_anchor}-{cell.benchmark_id}"
        )
        cell_ci = (
            f"{cell.ci_low:.3f}-{cell.ci_high:.3f}"
            if cell.ci_available and cell.ci_low is not None
            else "n/a"
        )
        benchmark_rows.append(
            "<tr>"
            f"<td>{redacted_esc(cell.benchmark_id, extra_secrets=extra_secrets)}</td>"
            f"<td>{redacted_esc(cell.category, extra_secrets=extra_secrets)}</td>"
            f"<td>{cell_score}</td>"
            f"<td>{esc(cell_ci)}</td>"
            f"<td>{cell.n_graded}/{cell.n_total}</td>"
            f"<td>{cell.n_missing}</td>"
            "</tr>"
        )
    predicted_items: list[str] = []
    for entry in endpoint.predicted:
        candidate = redact_text(str(entry.get("candidate_id", "")), extra_secrets=extra_secrets)
        similarity = entry.get("similarity")
        signals = entry.get("evidence_signals", [])
        note = entry.get("note")
        predicted_items.append(
            f"<li>{esc(candidate)} similarity {esc(similarity)} "
            f"signals {esc(list(signals))}"
            + (f" note {redacted_esc(note, extra_secrets=extra_secrets)}" if note else "")
            + "</li>"
        )
    predicted_section = (
        "<h3>Predicted identity (similarity, not probability)</h3>"
        + ("<ul>" + "".join(predicted_items) + "</ul>" if predicted_items else "<p>unknown</p>")
        + "<p>Similarities are not probabilities and never enter a score.</p>"
    )
    reveal_text = endpoint.reveal_label if endpoint.reveal_label is not None else "none recorded"
    reveal_section = (
        "<h3>Revealed identity (official, separate)</h3>"
        f"<p>{redacted_esc(reveal_text, extra_secrets=extra_secrets)}</p>"
        "<p>The reveal is recorded separately from any earlier prediction; "
        "a later reveal does not rewrite history.</p>"
    )
    stale_note = (
        "<p><strong>Stale evidence</strong> for this observation.</p>" if endpoint.stale else ""
    )
    body = (
        f"<p>Model {redacted_esc(endpoint.alias, extra_secrets=extra_secrets)} "
        f"endpoint {redacted_esc(endpoint.endpoint_id, extra_secrets=extra_secrets)} "
        f"route {redacted_esc(endpoint.route, extra_secrets=extra_secrets)}.</p>"
        f"<h2>Observation dates</h2><p>{esc(endpoint.observation_started)} "
        f"to {esc(endpoint.observation_ended)}</p>"
        f"<h2>Sample sizes</h2><p>{endpoint.n_graded} graded of {endpoint.n_total} "
        f"planned; {endpoint.n_correct} correct; {endpoint.n_missing} missing; "
        f"{endpoint.n_transport_failed} transport failures.</p>"
        f"<h2>Score</h2><p>Accuracy {linked} with 95% CI {esc(ci)}.</p>"
        f"<h2>Costs</h2><p>Billed {esc(endpoint.cost_billed)} spent "
        f"{esc(endpoint.cost_spent)}; input tokens {esc(endpoint.input_tokens)} "
        f"output tokens {esc(endpoint.output_tokens)}.</p>"
        f"<h2>Missing coverage</h2><p>Coverage {esc(endpoint.coverage)}; "
        f"missing {endpoint.n_missing} of {endpoint.n_total}.</p>"
        "<h2>Identity resolution</h2>"
        f"<p>State {esc(endpoint.identity_state)}; "
        f"abstention {esc(endpoint.abstention_reason)}.</p>"
        f"{predicted_section}{reveal_section}{stale_note}"
        "<h2>Benchmarks</h2><table><thead><tr><th>benchmark</th><th>category</th>"
        "<th>accuracy</th><th>95% CI</th><th>samples</th><th>missing</th>"
        "</tr></thead><tbody>" + "".join(benchmark_rows) + "</tbody></table>"
        '<p><a href="index.html">Leaderboard</a> | '
        '<a href="provenance.html">Provenance</a></p>'
    )
    return _page_shell(f"Model {endpoint.alias}", body)


def render_comparison(data: ReportData, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Pairwise comparisons with tied/uncertain handling."""
    rows: list[str] = []
    for item in data.comparisons:
        if not item.available:
            verdict_text = "uncertain (too few shared tasks)"
            diff = "n/a"
            ci = "n/a"
        elif item.verdict == "tied_or_uncertain":
            verdict_text = "tied_or_uncertain: no significant difference"
            diff = f"{item.observed_difference:.3f}" if item.observed_difference else "n/a"
            ci = f"{item.ci_low:.3f}-{item.ci_high:.3f}"
        elif item.verdict == "first_leads":
            verdict_text = f"{item.first_id} leads"
            diff = f"{item.observed_difference:.3f}"
            ci = f"{item.ci_low:.3f}-{item.ci_high:.3f}"
        elif item.verdict == "second_leads":
            verdict_text = f"{item.second_id} leads"
            diff = f"{item.observed_difference:.3f}"
            ci = f"{item.ci_low:.3f}-{item.ci_high:.3f}"
        else:
            verdict_text = "unavailable"
            diff = "n/a"
            ci = "n/a"
        linked = _score_link(diff, f"comparison-{item.first_id}-{item.second_id}")
        rows.append(
            "<tr>"
            f"<td>{redacted_esc(item.first_id, extra_secrets=extra_secrets)}</td>"
            f"<td>{redacted_esc(item.second_id, extra_secrets=extra_secrets)}</td>"
            f"<td>{redacted_esc(item.benchmark_id, extra_secrets=extra_secrets)}</td>"
            f"<td>{item.n_shared_tasks}</td>"
            f"<td>{linked}</td>"
            f"<td>{esc(ci)}</td>"
            f"<td>{redacted_esc(verdict_text, extra_secrets=extra_secrets)}</td>"
            "</tr>"
        )
    body = (
        "<h2>Comparisons on shared declared items</h2>"
        "<table><thead><tr><th>first</th><th>second</th><th>benchmark</th>"
        "<th>shared tasks</th><th>difference</th><th>95% CI</th><th>verdict</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<p><a href="index.html">Leaderboard</a> | '
        '<a href="provenance.html">Provenance</a></p>'
    )
    return _page_shell("Comparisons", body)


def render_history(data: ReportData, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Observation history with stale flags and separate reveal column."""
    rows: list[str] = []
    for endpoint in data.endpoints:
        stale_text = "stale" if endpoint.stale else "fresh"
        reveal = endpoint.reveal_label if endpoint.reveal_label is not None else "none"
        rows.append(
            "<tr>"
            f"<td>{redacted_esc(data.campaign_id, extra_secrets=extra_secrets)}</td>"
            f"<td>{redacted_esc(endpoint.alias, extra_secrets=extra_secrets)}</td>"
            f"<td>{esc(endpoint.observation_started)} to {esc(endpoint.observation_ended)}</td>"
            f"<td>{endpoint.n_graded}/{endpoint.n_total}</td>"
            f"<td>{esc(data.grade_digest)}</td>"
            f"<td>{esc(stale_text)}</td>"
            f"<td>{redacted_esc(endpoint.identity_state, extra_secrets=extra_secrets)}</td>"
            f"<td>{redacted_esc(reveal, extra_secrets=extra_secrets)}</td>"
            "</tr>"
        )
    body = (
        "<h2>History</h2>"
        "<table><thead><tr><th>campaign</th><th>model</th><th>window</th>"
        "<th>samples</th><th>grade digest</th><th>freshness</th>"
        "<th>predicted state</th><th>revealed label (separate)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p>Prediction and reveal are separate columns; a reveal never rewrites a prediction.</p>"
        '<p><a href="index.html">Leaderboard</a></p>'
    )
    return _page_shell("History", body)


def render_provenance(data: ReportData, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Every score's reproducible provenance in one place."""
    endpoint_rows: list[str] = []
    for endpoint in data.endpoints:
        endpoint_rows.append(
            f'<li id="{esc(endpoint.provenance_anchor)}">'
            f"{redacted_esc(endpoint.endpoint_id, extra_secrets=extra_secrets)} "
            f"accuracy {_score_text(endpoint.accuracy)} "
            f"manifest {esc(data.manifest_hash)} "
            f"grade {esc(data.grade_digest)}</li>"
        )
        for cell in endpoint.benchmarks:
            endpoint_rows.append(
                f'<li id="{esc(endpoint.provenance_anchor)}-{esc(cell.benchmark_id)}">'
                f"{redacted_esc(endpoint.endpoint_id, extra_secrets=extra_secrets)} "
                f"{redacted_esc(cell.benchmark_id, extra_secrets=extra_secrets)} "
                f"accuracy {_score_text(cell.accuracy)} "
                f"manifest {esc(data.manifest_hash)}</li>"
            )
    source_dir = str(data.provenance.get("artifacts_dir", ""))
    source_esc = redacted_esc(source_dir, extra_secrets=extra_secrets)
    body = (
        f'<p id="manifest">Manifest hash {esc(data.manifest_hash)} '
        f"score version {esc(data.score_version)} seed {data.seed}.</p>"
        f'<p id="grades">Grade digest {esc(data.grade_digest)} '
        f"plan digest {esc(data.plan_digest)} "
        f"events {data.event_count} accepted {data.accepted_samples}.</p>"
        f'<p id="ledger">Ledger {esc(json.dumps(data.ledger, sort_keys=True))}</p>'
        "<h2>Scores</h2><ul>" + "".join(endpoint_rows) + "</ul>"
        "<h2>Artifacts</h2>"
        f"<p>Source {source_esc}</p>"
        '<p><a href="index.html">Leaderboard</a></p>'
    )
    return _page_shell("Provenance", body)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _report_rows(data: ReportData) -> tuple[list[dict[str, Any]], list[str]]:
    columns = [
        "campaign_id",
        "endpoint_id",
        "alias",
        "benchmark_id",
        "category",
        "n_total",
        "n_graded",
        "n_correct",
        "n_missing",
        "accuracy",
        "ci_low",
        "ci_high",
        "identity_state",
        "reveal_label",
    ]
    rows: list[dict[str, Any]] = []
    for endpoint in data.endpoints:
        if not endpoint.benchmarks:
            rows.append(
                {
                    "accuracy": endpoint.accuracy,
                    "alias": endpoint.alias,
                    "benchmark_id": "",
                    "campaign_id": data.campaign_id,
                    "category": "",
                    "ci_high": endpoint.ci_high,
                    "ci_low": endpoint.ci_low,
                    "endpoint_id": endpoint.endpoint_id,
                    "identity_state": endpoint.identity_state,
                    "n_correct": endpoint.n_correct,
                    "n_graded": endpoint.n_graded,
                    "n_missing": endpoint.n_missing,
                    "n_total": endpoint.n_total,
                    "reveal_label": endpoint.reveal_label,
                }
            )
        for cell in endpoint.benchmarks:
            rows.append(
                {
                    "accuracy": cell.accuracy,
                    "alias": endpoint.alias,
                    "benchmark_id": cell.benchmark_id,
                    "campaign_id": data.campaign_id,
                    "category": cell.category,
                    "ci_high": cell.ci_high,
                    "ci_low": cell.ci_low,
                    "endpoint_id": endpoint.endpoint_id,
                    "identity_state": endpoint.identity_state,
                    "n_correct": cell.n_correct,
                    "n_graded": cell.n_graded,
                    "n_missing": cell.n_missing,
                    "n_total": cell.n_total,
                    "reveal_label": endpoint.reveal_label,
                }
            )
    return rows, columns


def build_report(
    artifacts_dir: Path,
    output_dir: Path,
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build the static report offline into ``output_dir``.

    Writes ``index.html``, one ``endpoint-<slug>.html`` per endpoint,
    ``comparison.html``, ``history.html``, ``provenance.html``,
    ``report.json`` and ``report.csv``. Returns a JSON-serializable summary
    with counts matching the source artifacts. Raises ``ValueError`` for a
    non-artifact input.
    """
    data = build_report_data(Path(artifacts_dir), extra_secrets=extra_secrets)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    pages: dict[str, str] = {
        "index.html": render_index(data, extra_secrets=extra_secrets),
        "comparison.html": render_comparison(data, extra_secrets=extra_secrets),
        "history.html": render_history(data, extra_secrets=extra_secrets),
        "provenance.html": render_provenance(data, extra_secrets=extra_secrets),
    }
    for endpoint in data.endpoints:
        pages[f"endpoint-{endpoint.slug}.html"] = render_endpoint(
            data, endpoint, extra_secrets=extra_secrets
        )
    for name, text in pages.items():
        target = _resolve_inside(out, name)
        redacted = redact_text(text, extra_secrets=extra_secrets)
        write_text_atomic(target, redacted)
        files.append(name)

    payload = data.to_dict()
    json_text = redact_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
        extra_secrets=extra_secrets,
    )
    json_target = _resolve_inside(out, REPORT_JSON_NAME)
    write_text_atomic(json_target, json_text + "\n")
    files.append(REPORT_JSON_NAME)

    rows, columns = _report_rows(data)
    # Reuse the traversal-safe, defusing CSV writer via inline logic that
    # mirrors exports.rows_to_csv (kept here to avoid a circular import at
    # module scope; behavior is identical).
    import csv as _csv
    import io as _io

    buffer = _io.StringIO()
    writer = _csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        safe: dict[str, Any] = {}
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, str) and value == ""):
                safe[column] = ""
            elif isinstance(value, str):
                safe[column] = sanitize_csv_cell(value, extra_secrets=extra_secrets)
            else:
                safe[column] = sanitize_csv_cell(str(value), extra_secrets=extra_secrets)
        writer.writerow(safe)
    csv_target = _resolve_inside(out, REPORT_CSV_NAME)
    write_text_atomic(csv_target, buffer.getvalue())
    files.append(REPORT_CSV_NAME)

    # A redacted manifest copy anchors provenance without leaking secrets.
    manifest_src = Path(artifacts_dir) / "manifest.json"
    if manifest_src.is_file():
        manifest_text = redact_text(
            manifest_src.read_text(encoding="utf-8"), extra_secrets=extra_secrets
        )
        manifest_target = _resolve_inside(out, "manifest.json")
        write_text_atomic(manifest_target, manifest_text)
        files.append("manifest.json")

    return {
        "accepted_samples": data.accepted_samples,
        "artifacts_dir": str(artifacts_dir),
        "campaign_id": data.campaign_id,
        "endpoints": len(data.endpoints),
        "event_count": data.event_count,
        "files": sorted(files),
        "grade_digest": data.grade_digest,
        "manifest_hash": data.manifest_hash,
        "output_dir": str(out),
        "report_version": REPORT_VERSION,
        "status": "ok",
    }


def serve_report(
    output_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a built report over HTTP on loopback only.

    ``host`` must be a loopback address; anything else raises ``ValueError``
    because deployment is a separate authorized action. Blocks serving.
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to serve on non-loopback host {host!r}; "
            "deployment is a separate authorized action"
        )
    import functools
    import http.server

    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"no built report at {output_dir}")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer((host, port), handler) as server:
        server.serve_forever()


__all__ = [
    "CI_CONFIDENCE",
    "CI_RE_SAMPLES",
    "LOOPBACK_HOSTS",
    "REPORT_VERSION",
    "STALE_AFTER_DAYS",
    "BenchmarkCell",
    "EndpointSummary",
    "PairwiseComparison",
    "ReportData",
    "build_report",
    "build_report_data",
    "esc",
    "redacted_esc",
    "render_comparison",
    "render_endpoint",
    "render_history",
    "render_index",
    "render_provenance",
    "safe_slug",
    "serve_report",
]
