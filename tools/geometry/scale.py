"""Metric scale estimation tool slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec
from peps.tools.geometry.reconstruction import ReconstructionResult


@dataclass(slots=True)
class MetricScaleResult:
    scale_factor: float
    selected_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        return f"MetricScaleResult: scale_factor={self.scale_factor:.4f}"


class EstimateScaleTool(BaseTool):
    spec = ToolSpec(
        name="estimate_scale",
        description="Estimate metric scale factor for a relative 3D reconstruction.",
        input_schema={
            "type": "object",
            "required": ["reconstruction"],
            "properties": {
                "reconstruction": {"type": "object"},
                "known_reference_distance": {"type": "number"},
                "reconstruction_reference_distance": {"type": "number"},
            },
        },
        output_schema={"type": "object", "class": "MetricScaleResult"},
        model_requirements=[
            LocalModelRequirement(
                name="MoGe-2",
                required_for="metric depth alignment and scale estimation",
                local_path_env="PEPS_MOGE_MODEL_DIR",
                package="moge",
                download_hint="Download MoGe/MoGe-2 weights locally and set PEPS_MOGE_MODEL_DIR.",
                required=False,
            )
        ],
        tags=["geometry", "scale"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        reconstruction: ReconstructionResult,
        known_reference_distance: float | None = None,
        reconstruction_reference_distance: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if known_reference_distance is not None and reconstruction_reference_distance:
            scale_factor = float(known_reference_distance) / float(reconstruction_reference_distance)
            note = "Scale computed from provided reference distance."
        else:
            scale_factor = 1.0
            note = "Placeholder scale factor. Configure MoGe backend for metric scale."
        result = MetricScaleResult(
            scale_factor=scale_factor,
            selected_indices=list(range(len(reconstruction.camera_poses))),
            metadata={"note": note},
        )
        return self.success(result)

