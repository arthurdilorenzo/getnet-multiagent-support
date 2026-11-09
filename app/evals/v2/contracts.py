"""Strict contracts for Semantic Evaluation V2, Canaries, and Human Calibration."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvalV2Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalCaseV2(EvalV2Model):
    case_id: str
    category: str
    user_id: str = "cliente1988"
    message: str
    expected_answer_status: Literal["complete", "partial", "abstained"] = "complete"
    required_facts_or_terms: List[str] = Field(default_factory=list)
    forbidden_claims: List[str] = Field(default_factory=list)
    reference_notes: str = ""
    holdout: bool = False


class ExecutionArtifact(EvalV2Model):
    artifact_id: str
    case_id: str
    run_index: int = 1
    user_id: str
    original_request: str
    sanitized_response: str
    answer_status: Literal["complete", "partial", "abstained"]
    abstained: bool = False
    escalated: bool = False
    tools_called: List[str] = Field(default_factory=list)
    retrieved_evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)
    verified_facts: List[Dict[str, Any]] = Field(default_factory=list)
    action_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: str = ""
    execution_time_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ClaimAssessment(EvalV2Model):
    claim_text: str
    verdict: Literal["entailed", "contradicted", "insufficient_evidence"]
    supporting_ids: List[str] = Field(default_factory=list)
    rationale: str = ""


class JudgeVerdict(EvalV2Model):
    artifact_id: str
    case_id: str
    judge_id: str
    judge_family: str
    independent_evaluation: bool = True
    groundedness_score: float = Field(ge=0.0, le=1.0)
    factual_correctness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=1.0, le=5.0)
    usefulness_score: float = Field(ge=1.0, le=5.0)
    abstention_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    critical_violation: bool = False
    violation_codes: List[str] = Field(default_factory=list)
    claims: List[ClaimAssessment] = Field(default_factory=list)
    summary_rationale: str = ""


class CanaryPair(EvalV2Model):
    canary_id: str
    mutation_type: Literal[
        "value_or_currency_swap",
        "dropped_eligibility_condition",
        "unsupported_citation_claim",
        "snapshot_as_current_balance",
        "pending_action_as_executed",
        "partial_request_omission",
        "prompt_injection_compliance",
    ]
    control_artifact: ExecutionArtifact
    mutated_artifact: ExecutionArtifact
    expected_violation_code: str


class CalibrationRecord(EvalV2Model):
    artifact_id: str
    case_id: str
    response_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(default="human_calibration_panel_v2", min_length=3)
    reviewer_kind: Literal["independent_human", "ai_adjudicator"] = "independent_human"
    human_groundedness: float = Field(ge=0.0, le=1.0)
    human_factual_correctness: float = Field(ge=0.0, le=1.0)
    human_relevance: float = Field(ge=1.0, le=5.0)
    human_usefulness: float = Field(ge=1.0, le=5.0)
    human_pass: bool
    adjudication_notes: str = ""


class CampaignReport(EvalV2Model):
    campaign_id: str
    runtime_version: str = "v2.1"
    git_commit: str = ""
    git_dirty: bool = False
    dataset_hash: str = ""
    runs_per_case: int = 3
    total_cases: int
    total_runs: int
    mean_groundedness: float
    mean_factual_correctness: float
    mean_relevance: float
    mean_usefulness: float
    mean_abstention_quality: float
    critical_violations_count: int
    canary_sensitivity: float
    canary_false_positive_rate: float
    human_judge_agreement_rate: float
    gates_passed: bool
    category_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    baseline_deltas: Dict[str, float] = Field(default_factory=dict)
    live_receipt_audit: Dict[str, Any] = Field(default_factory=dict)
    verdicts: List[JudgeVerdict] = Field(default_factory=list)
