"""Campaign Runner, Canary Evaluator, and Human Calibration Aggregator for Semantic Evaluation V2."""
import hashlib
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.evals.v2.contracts import (
    CalibrationRecord,
    CampaignReport,
    CanaryPair,
    EvalCaseV2,
    ExecutionArtifact,
    JudgeVerdict,
)
from app.evals.v2.judge import IndependentSemanticJudge
from app.schemas import ChatResponse
from app.security.auth import SecurityContext


class SemanticEvalCampaignRunner:
    """Executes multi-run V2 evaluation campaigns, canary sensitivity checks, and human calibration."""

    def __init__(
        self,
        judge: Optional[IndependentSemanticJudge] = None,
        git_commit: str = "HEAD",
        git_dirty: bool = False,
        runtime_version: str = "v2.1",
    ) -> None:
        self.judge = judge or IndependentSemanticJudge()
        self.git_commit = git_commit
        self.git_dirty = git_dirty
        self.runtime_version = runtime_version

    @staticmethod
    def build_artifact_from_response(
        case: EvalCaseV2,
        response: ChatResponse,
        run_index: int = 1,
    ) -> ExecutionArtifact:
        spans = [
            e for e in (response.evidence or [])
            if isinstance(e, dict) and "span_id" in e
        ]
        facts = [
            e for e in (response.evidence or [])
            if isinstance(e, dict) and "fact_id" in e
        ]
        return ExecutionArtifact(
            artifact_id=f"art_{case.case_id}_r{run_index}",
            case_id=case.case_id,
            run_index=run_index,
            user_id=case.user_id,
            original_request=case.message,
            sanitized_response=response.response,
            answer_status=response.answer_status,
            abstained=response.abstained,
            escalated=response.escalated,
            tools_called=list(response.tools_called or []),
            retrieved_evidence_spans=spans,
            verified_facts=facts,
            action_proposals=list(response.action_proposals or []),
            trace_id=response.trace_id,
            execution_time_ms=response.execution_time_ms,
            total_tokens=response.token_usage.total_tokens if response.token_usage else 0,
            estimated_cost_usd=response.token_usage.estimated_cost_usd if response.token_usage else 0.0,
        )

    def evaluate_canaries(
        self,
        canary_pairs: List[CanaryPair],
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        if not canary_pairs:
            # Fail closed when no canary pairs are supplied
            return 0.0, 1.0, []

        detected = 0
        false_positives = 0
        details: List[Dict[str, Any]] = []

        for pair in canary_pairs:
            ctrl_verdict = self.judge.evaluate(pair.control_artifact)
            mut_verdict = self.judge.evaluate(pair.mutated_artifact)

            ctrl_flagged = (
                ctrl_verdict.critical_violation
                or bool(ctrl_verdict.violation_codes)
                or ctrl_verdict.groundedness_score < 0.9
                or ctrl_verdict.factual_correctness_score < 0.9
            )
            mut_flagged = (
                pair.expected_violation_code in mut_verdict.violation_codes
                or mut_verdict.critical_violation
                or mut_verdict.groundedness_score < 0.8
                or mut_verdict.factual_correctness_score < 0.8
            )
            if ctrl_flagged:
                false_positives += 1
            if mut_flagged:
                detected += 1

            details.append(
                {
                    "canary_id": pair.canary_id,
                    "mutation_type": pair.mutation_type,
                    "expected_violation_code": pair.expected_violation_code,
                    "observed_violations": mut_verdict.violation_codes,
                    "control_clean": not ctrl_flagged,
                    "mutation_detected": mut_flagged,
                }
            )

        sensitivity = round(detected / len(canary_pairs), 4)
        fp_rate = round(false_positives / len(canary_pairs), 4)
        return sensitivity, fp_rate, details

    def evaluate_calibration(
        self,
        artifacts_by_id: Dict[str, ExecutionArtifact],
        calibration_records: List[CalibrationRecord],
    ) -> float:
        if not calibration_records:
            # Fail closed when no calibration records are provided
            return 0.0
        agreements = 0
        total_records = len(calibration_records)
        import re
        for rec in calibration_records:
            # Missing or non-hex response_sha256 fails closed
            if not rec.response_sha256 or not re.fullmatch(r"[0-9a-f]{64}", rec.response_sha256):
                continue
            art = artifacts_by_id.get(rec.artifact_id)
            if art is None:
                # Missing artifact stays in denominator (total_records) and counts as 0 agreement
                continue
            normalized_resp = re.sub(r"\bPROP-[A-F0-9]+\b", "PROP-<ID>", art.sanitized_response or "")
            normalized_resp = re.sub(r"\b20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b", "<TIMESTAMP>", normalized_resp)
            actual_sha = hashlib.sha256(normalized_resp.encode("utf-8")).hexdigest()
            if actual_sha != rec.response_sha256:
                # Cryptographic hash mismatch invalidates the calibration record
                continue
            verdict = self.judge.evaluate(art)
            judge_pass = (
                not verdict.critical_violation
                and not verdict.violation_codes
                and verdict.groundedness_score >= 0.9
                and verdict.factual_correctness_score >= 0.9
            )
            if judge_pass == rec.human_pass:
                agreements += 1
        return round(agreements / total_records, 4)

    def run_campaign(
        self,
        cases: List[EvalCaseV2],
        execute_fn: Callable[[EvalCaseV2, int], ChatResponse],
        *,
        runs_per_case: int = 3,
        canary_pairs: Optional[List[CanaryPair]] = None,
        calibration_records: Optional[List[CalibrationRecord]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        live_receipt_audit: Optional[Dict[str, Any]] = None,
    ) -> CampaignReport:
        verdicts: List[JudgeVerdict] = []
        artifacts_by_id: Dict[str, ExecutionArtifact] = {}
        cat_buckets: Dict[str, List[JudgeVerdict]] = {}

        for case in cases:
            for r_idx in range(1, runs_per_case + 1):
                resp = execute_fn(case, r_idx)
                artifact = self.build_artifact_from_response(case, resp, run_index=r_idx)
                artifacts_by_id[artifact.artifact_id] = artifact
                verdict = self.judge.evaluate(artifact, reference=case)
                verdicts.append(verdict)
                cat_buckets.setdefault(case.category, []).append(verdict)

        for pair in canary_pairs or []:
            artifacts_by_id[pair.control_artifact.artifact_id] = pair.control_artifact
            artifacts_by_id[pair.mutated_artifact.artifact_id] = pair.mutated_artifact

        canary_sens, canary_fp, _ = self.evaluate_canaries(canary_pairs or [])
        cal_agreement = self.evaluate_calibration(artifacts_by_id, calibration_records or [])

        n = max(1, len(verdicts))
        mean_ground = round(sum(v.groundedness_score for v in verdicts) / n, 4)
        mean_fact = round(sum(v.factual_correctness_score for v in verdicts) / n, 4)
        mean_rel = round(sum(v.relevance_score for v in verdicts) / n, 2)
        mean_use = round(sum(v.usefulness_score for v in verdicts) / n, 2)
        mean_abs = round(sum(v.abstention_quality_score for v in verdicts) / n, 4)
        crit_count = sum(1 for v in verdicts if v.critical_violation)

        category_breakdown: Dict[str, Dict[str, float]] = {}
        for cat, cat_verdicts in cat_buckets.items():
            cn = max(1, len(cat_verdicts))
            category_breakdown[cat] = {
                "runs": float(len(cat_verdicts)),
                "groundedness": round(sum(v.groundedness_score for v in cat_verdicts) / cn, 4),
                "factual_correctness": round(sum(v.factual_correctness_score for v in cat_verdicts) / cn, 4),
                "relevance": round(sum(v.relevance_score for v in cat_verdicts) / cn, 2),
                "usefulness": round(sum(v.usefulness_score for v in cat_verdicts) / cn, 2),
            }

        baseline_deltas: Dict[str, float] = {}
        if baseline_metrics:
            baseline_deltas = {
                "groundedness_delta": round(mean_ground - baseline_metrics.get("mean_groundedness", mean_ground), 4),
                "factual_correctness_delta": round(mean_fact - baseline_metrics.get("mean_factual_correctness", mean_fact), 4),
                "relevance_delta": round(mean_rel - baseline_metrics.get("mean_relevance", mean_rel), 2),
                "usefulness_delta": round(mean_use - baseline_metrics.get("mean_usefulness", mean_use), 2),
            }

        live_gate_ok = True
        if live_receipt_audit is not None:
            live_gate_ok = (
                int(live_receipt_audit.get("evaluated", 0)) > 0
                and float(live_receipt_audit.get("agreement_rate", 0.0)) >= 0.90
            )

        gates_passed = (
            crit_count == 0
            and mean_ground >= 0.95
            and mean_fact >= 0.95
            and mean_rel >= 4.0
            and mean_use >= 4.0
            and canary_sens >= 0.95
            and canary_fp <= 0.05
            and cal_agreement >= 0.90
            and live_gate_ok
        )

        dataset_hash = hashlib.sha256(
            json.dumps([c.model_dump() for c in cases], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

        return CampaignReport(
            campaign_id=f"camp_v2_{uuid.uuid4().hex[:10]}",
            runtime_version=self.runtime_version,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            dataset_hash=dataset_hash,
            runs_per_case=runs_per_case,
            total_cases=len(cases),
            total_runs=len(verdicts),
            mean_groundedness=mean_ground,
            mean_factual_correctness=mean_fact,
            mean_relevance=mean_rel,
            mean_usefulness=mean_use,
            mean_abstention_quality=mean_abs,
            critical_violations_count=crit_count,
            canary_sensitivity=canary_sens,
            canary_false_positive_rate=canary_fp,
            human_judge_agreement_rate=cal_agreement,
            gates_passed=gates_passed,
            category_breakdown=category_breakdown,
            baseline_deltas=baseline_deltas,
            live_receipt_audit=live_receipt_audit or {},
            verdicts=verdicts,
        )
