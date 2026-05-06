"""Verifier feedback handling for PEPS refinement rounds."""

from __future__ import annotations

from dataclasses import dataclass, field

from peps.core.enums import FeedbackType
from peps.core.trace import ExecutionTrace, VerificationResult
from peps.core.types import PrimitiveHypothesis
from peps.schema.candidate_pool import CandidatePrimitivePool


@dataclass(slots=True)
class ParserRefinementContext:
    """Context passed into the next Parser invocation."""

    feedback: str | None = None
    feedback_type: FeedbackType | None = None
    candidate_hypotheses: list[PrimitiveHypothesis | object] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RefinementManager:
    """Convert Verifier feedback into Parser refinement context."""

    def __init__(
        self,
        candidate_pool: CandidatePrimitivePool | None = None,
        *,
        include_active_candidates: bool = True,
    ) -> None:
        self.candidate_pool = candidate_pool
        self.include_active_candidates = include_active_candidates

    def prepare_next_parser_context(
        self,
        verification: VerificationResult,
        *,
        query: str,
        trace: ExecutionTrace,
    ) -> ParserRefinementContext:
        if verification.feedback_type is FeedbackType.NONE:
            return ParserRefinementContext()

        candidates: list[PrimitiveHypothesis | object] = []
        if verification.feedback_type is FeedbackType.PRIMITIVE_GAP:
            for hypothesis in verification.primitive_gap_hypotheses:
                candidates.append(hypothesis)
                if self.candidate_pool is not None:
                    self.candidate_pool.add_hypothesis(
                        hypothesis,
                        query=query,
                        trace_id=trace.trace_id,
                    )
            if self.candidate_pool is not None and self.include_active_candidates:
                candidates.extend(self.candidate_pool.active())

        return ParserRefinementContext(
            feedback=verification.feedback,
            feedback_type=verification.feedback_type,
            candidate_hypotheses=candidates,
            metadata={
                "trace_id": trace.trace_id,
                "quality_score": verification.quality_score,
                "missing_slots": verification.missing_slots,
            },
        )

