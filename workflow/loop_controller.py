"""Loop control for PEPS refinement rounds."""

from __future__ import annotations

from dataclasses import dataclass

from peps.core.trace import VerificationResult
from peps.workflow.router import RouteDecision, WorkflowRoute, WorkflowRouter


@dataclass(slots=True)
class WorkflowLoopConfig:
    max_rounds: int = 3
    min_accept_score: float = 0.0

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("WorkflowLoopConfig.max_rounds must be >= 1")
        if not 0.0 <= self.min_accept_score <= 1.0:
            raise ValueError("WorkflowLoopConfig.min_accept_score must be in [0, 1]")


@dataclass(slots=True)
class LoopDecision:
    route: WorkflowRoute
    reason: str
    round_index: int
    can_refine: bool
    accepted: bool = False
    exhausted: bool = False


class LoopController:
    """Decide whether the PEPS workflow should accept, refine, or stop."""

    def __init__(
        self,
        config: WorkflowLoopConfig | None = None,
        router: WorkflowRouter | None = None,
    ) -> None:
        self.config = config or WorkflowLoopConfig()
        self.router = router or WorkflowRouter()

    def can_start_round(self, round_index: int) -> bool:
        return round_index < self.config.max_rounds

    def can_refine_after(self, round_index: int) -> bool:
        return round_index + 1 < self.config.max_rounds

    def evaluate(
        self,
        verification: VerificationResult | None,
        *,
        round_index: int,
    ) -> LoopDecision:
        can_refine = self.can_refine_after(round_index)
        route = self.router.route_after_verification(
            verification,
            can_refine=can_refine,
            min_accept_score=self.config.min_accept_score,
        )
        return LoopDecision(
            route=route.route,
            reason=route.reason,
            round_index=round_index,
            can_refine=can_refine,
            accepted=route.route is WorkflowRoute.ACCEPT,
            exhausted=route.route is WorkflowRoute.REJECT,
        )

