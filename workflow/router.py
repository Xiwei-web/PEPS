"""Routing decisions for PEPS workflow transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from peps.core.enums import FeedbackType, VerificationDecision
from peps.core.trace import VerificationResult


class WorkflowRoute(StrEnum):
    PARSE = "parse"
    EXECUTE = "execute"
    CODE = "code"
    VERIFY = "verify"
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"
    STOP = "stop"


@dataclass(slots=True)
class RouteDecision:
    route: WorkflowRoute
    reason: str
    feedback_type: FeedbackType = FeedbackType.NONE


class WorkflowRouter:
    """Rule-based router used by the loop controller."""

    def route_after_verification(
        self,
        verification: VerificationResult | None,
        *,
        can_refine: bool,
        min_accept_score: float = 0.0,
    ) -> RouteDecision:
        if verification is None:
            return RouteDecision(WorkflowRoute.VERIFY, "No verification result yet.")
        if (
            verification.decision is VerificationDecision.ACCEPT
            and verification.quality_score >= min_accept_score
        ):
            return RouteDecision(
                WorkflowRoute.ACCEPT,
                f"Verifier accepted with score {verification.quality_score:.3f}.",
                verification.feedback_type,
            )
        if verification.decision is VerificationDecision.ACCEPT:
            reason = (
                f"Verifier accepted but score {verification.quality_score:.3f} "
                f"is below threshold {min_accept_score:.3f}."
            )
            return RouteDecision(
                WorkflowRoute.REFINE if can_refine else WorkflowRoute.REJECT,
                reason,
                verification.feedback_type,
            )
        if can_refine:
            return RouteDecision(
                WorkflowRoute.REFINE,
                f"Verifier rejected with {verification.feedback_type.value} feedback.",
                verification.feedback_type,
            )
        return RouteDecision(
            WorkflowRoute.REJECT,
            "Verifier rejected and loop budget is exhausted.",
            verification.feedback_type,
        )

