"""Runtime wrapper and PEPS code tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.core.io import to_plain_data
from peps.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from peps.tools.code.sandbox import CodeSandbox, SandboxConfig, SandboxExecutionResult
from peps.tools.workspace import ToolWorkspace


@dataclass(slots=True)
class CodeRuntimeConfig:
    """Configuration for deterministic code execution."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


class CodeRuntime:
    """Execute generated deterministic code over PEPS workspace variables."""

    def __init__(self, config: CodeRuntimeConfig | None = None) -> None:
        self.config = config or CodeRuntimeConfig()
        self.sandbox = CodeSandbox(self.config.sandbox)

    def execute(
        self,
        code: str,
        workspace: ToolWorkspace | dict[str, Any],
    ) -> SandboxExecutionResult:
        if isinstance(workspace, ToolWorkspace):
            workspace_data = workspace.raw_values()
        else:
            workspace_data = workspace
        return self.sandbox.run(code, to_plain_data(workspace_data))


class CodeTool(BaseTool):
    """ToolRegistry-compatible code execution tool."""

    spec = ToolSpec(
        name="code",
        description="Execute deterministic Python code over already-acquired workspace variables.",
        input_schema={
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string"},
                "workspace": {"type": "object"},
            },
        },
        output_schema={"type": "object", "class": "SandboxExecutionResult"},
        tags=["code", "deterministic_compute"],
    )

    def __init__(self, runtime: CodeRuntime | None = None) -> None:
        super().__init__()
        self.runtime = runtime or CodeRuntime()

    async def _arun(
        self,
        *,
        context: ToolContext,
        code: str,
        workspace: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        workspace_obj = workspace if workspace is not None else context.workspace
        if workspace_obj is None:
            workspace_obj = {}
        result = self.runtime.execute(code, workspace_obj)
        if not result.ok:
            return self.error(result.error or "code execution failed", result=to_plain_data(result))
        return self.success(result)

