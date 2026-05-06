"""Registry and dispatcher for PEPS tools."""

from __future__ import annotations

import asyncio
from typing import Any

from peps.core.enums import ValueSource
from peps.core.exceptions import RegistryError
from peps.tools.base import BaseTool, ToolContext, ToolRequest, ToolResult, ToolSpec, UnavailableTool
from peps.tools.workspace import ToolWorkspace


class ToolRegistry:
    """Name-based registry that can dispatch Executor tool requests."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, *, overwrite: bool = False) -> BaseTool:
        if tool.name in self._tools and not overwrite:
            raise RegistryError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown tool: {name}. Available: {sorted(self._tools)}") from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> dict[str, ToolSpec]:
        return {name: tool.spec for name, tool in self._tools.items()}

    async def dispatch(
        self,
        request: ToolRequest,
        *,
        workspace: ToolWorkspace | None = None,
        context: ToolContext | None = None,
    ) -> ToolResult:
        workspace = workspace or ToolWorkspace()
        context = context or ToolContext(workspace=workspace)
        if context.workspace is None:
            context.workspace = workspace
        tool = self.get(request.tool_name)
        resolved_args = workspace.resolve(request.arguments)
        result = await tool.arun(context=context, **resolved_args)
        if result.ok and request.output_name:
            workspace.set(
                request.output_name,
                result.result,
                source=ValueSource.TOOL,
                metadata={
                    "tool_name": tool.name,
                    "call_id": request.call_id,
                    "fills": request.fills,
                },
            )
        return result

    def dispatch_sync(
        self,
        request: ToolRequest,
        *,
        workspace: ToolWorkspace | None = None,
        context: ToolContext | None = None,
    ) -> ToolResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dispatch(request, workspace=workspace, context=context))
        raise RuntimeError("dispatch_sync cannot be used inside a running event loop")


def build_default_tool_registry(
    *,
    include_code_tool: bool = True,
    include_code_placeholder: bool = False,
) -> ToolRegistry:
    """Build a registry containing PEPS's GCA-style eight API slots."""
    from peps.tools.geometry.detection import DetectionTool
    from peps.tools.geometry.motion import AnalyzeMotionTool
    from peps.tools.geometry.ocr import OCRTool
    from peps.tools.geometry.pose import PredictObjectPoseTool
    from peps.tools.geometry.projection import ProjectBoxTo3DPointsTool
    from peps.tools.geometry.reconstruction import ReconstructionTool
    from peps.tools.geometry.scale import EstimateScaleTool

    registry = ToolRegistry()
    registry.register(ReconstructionTool())
    registry.register(DetectionTool())
    registry.register(ProjectBoxTo3DPointsTool())
    registry.register(PredictObjectPoseTool())
    registry.register(EstimateScaleTool())
    registry.register(OCRTool())
    registry.register(AnalyzeMotionTool())
    if include_code_tool:
        from peps.tools.code.runtime import CodeTool

        registry.register(CodeTool())
    elif include_code_placeholder:
        registry.register(
            UnavailableTool(
                ToolSpec(
                    name="code",
                    description="Generate and execute deterministic Python code over acquired workspace variables. Implemented later by peps.tools.code.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    tags=["code", "deterministic_compute"],
                    is_placeholder=True,
                )
            )
        )
    return registry
