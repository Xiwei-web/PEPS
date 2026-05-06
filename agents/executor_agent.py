"""Executor Agent for PEPS.

The Executor plans and dispatches tool calls to acquire values for a fixed FESM
requirement set. It does not revise primitives or compute the final answer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from peps.core.enums import ToolCallStatus, TraceStage
from peps.core.exceptions import LLMOutputError, ValidationError
from peps.core.io import to_plain_data
from peps.core.trace import AgentRunRecord, ExecutionTrace, ToolCallRecord
from peps.core.types import FESMRequirementSet, QueryInput
from peps.llm.image_payload import ImagePayload
from peps.llm.openai_client import OpenAIClient, OpenAIResponse
from peps.prompts.executor_prompt import (
    EXECUTOR_PROMPT_VERSION,
    EXECUTOR_RESPONSE_JSON_SCHEMA,
    build_executor_system_prompt,
    build_executor_user_prompt,
)
from peps.tools.base import ToolContext, ToolRequest, ToolResult
from peps.tools.tool_registry import ToolRegistry, build_default_tool_registry
from peps.tools.workspace import ToolWorkspace


class ExecutorLLMClient(Protocol):
    """Minimal client interface required by ExecutorAgent."""

    def create_response(self, **kwargs: Any) -> OpenAIResponse:
        ...


@dataclass(slots=True)
class ExecutorAgentConfig:
    """Runtime settings for ExecutorAgent."""

    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    use_json_schema: bool = True
    strict_json_schema: bool = False
    validate_tool_calls: bool = True
    execute_tools: bool = True
    max_tool_calls: int = 12
    allow_code_tool: bool = False
    parallel_same_step: bool = True
    stop_on_tool_error: bool = True


@dataclass(slots=True)
class ExecutorToolCallPlan:
    """One planned tool call from the Executor LLM output."""

    step_id: int
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output_name: str | None = None
    fills: list[str] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, default_step_id: int) -> "ExecutorToolCallPlan":
        tool_name = data.get("tool") or data.get("tool_name")
        if not tool_name:
            raise ValidationError("Executor tool call missing 'tool' field")
        return cls(
            step_id=int(data.get("step_id", default_step_id)),
            tool_name=str(tool_name),
            arguments=dict(data.get("arguments", {})),
            output_name=data.get("output_name") or data.get("output_variable"),
            fills=[str(item) for item in data.get("fills", [])],
            rationale=data.get("rationale", ""),
            metadata=data.get("metadata", {}),
        )

    def to_request(self) -> ToolRequest:
        return ToolRequest(
            tool_name=self.tool_name,
            arguments=self.arguments,
            output_name=self.output_name,
            fills=self.fills,
            metadata={
                "step_id": self.step_id,
                "rationale": self.rationale,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class ExecutorAgentResult:
    """Structured result of one Executor invocation."""

    parsed_output: dict[str, Any]
    raw_output: str
    agent_run: AgentRunRecord
    tool_plan: list[ExecutorToolCallPlan] = field(default_factory=list)
    tool_records: list[ToolCallRecord] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    workspace: ToolWorkspace = field(default_factory=ToolWorkspace)


class ExecutorAgent:
    """Plan and execute tool calls for a fixed FESM requirement set."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        llm_client: ExecutorLLMClient | None = None,
        config: ExecutorAgentConfig | None = None,
    ) -> None:
        self.tool_registry = tool_registry or build_default_tool_registry()
        self.llm_client = llm_client or OpenAIClient.from_env()
        self.config = config or ExecutorAgentConfig()

    async def arun(
        self,
        requirements: FESMRequirementSet,
        *,
        query_input: QueryInput | None = None,
        workspace: ToolWorkspace | None = None,
        trace: ExecutionTrace | None = None,
        executor_feedback: str | None = None,
        model: str | None = None,
    ) -> ExecutorAgentResult:
        """Plan and optionally execute tool calls."""
        workspace = workspace or ToolWorkspace()
        system_prompt = build_executor_system_prompt(
            allow_code_tool=self.config.allow_code_tool
        )
        user_prompt = build_executor_user_prompt(
            requirements,
            self.tool_registry,
            query_input=query_input,
            workspace=workspace,
            executor_feedback=executor_feedback,
            allow_code_tool=self.config.allow_code_tool,
        )
        agent_run = AgentRunRecord(
            agent_name="ExecutorAgent",
            stage=TraceStage.EXECUTOR,
            prompt_name=EXECUTOR_PROMPT_VERSION,
            status=ToolCallStatus.RUNNING,
            input_summary={
                "query_id": query_input.query_id if query_input else None,
                "schema_version": requirements.schema_version,
                "num_requirements": len(requirements.all_instances()),
                "workspace_names": workspace.names(),
                "available_tools": self._available_tool_names(),
                "allow_code_tool": self.config.allow_code_tool,
            },
        )

        try:
            response = self.llm_client.create_response(
                system_prompt=system_prompt,
                user_text=user_prompt,
                images=[
                    ImagePayload.from_image_ref(image)
                    for image in (query_input.images if query_input else [])
                ],
                model=model or self.config.model,
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                text_format=self._text_format(),
            )
            parsed_output = response.json_object()
            tool_plan = self.parse_tool_plan(parsed_output)
        except Exception as exc:
            agent_run.mark_failed(str(exc))
            if trace is not None:
                trace.add_agent_run(agent_run)
            raise

        agent_run.raw_output = response.text
        agent_run.parsed_output = parsed_output
        agent_run.mark_succeeded()
        if trace is not None:
            trace.add_agent_run(agent_run)

        tool_records: list[ToolCallRecord] = []
        tool_results: list[ToolResult] = []
        if self.config.execute_tools:
            tool_records, tool_results = await self._execute_tool_plan(
                tool_plan,
                workspace=workspace,
                trace=trace,
                requirements=requirements,
                query_input=query_input,
            )

        return ExecutorAgentResult(
            parsed_output=parsed_output,
            raw_output=response.text,
            agent_run=agent_run,
            tool_plan=tool_plan,
            tool_records=tool_records,
            tool_results=tool_results,
            workspace=workspace,
        )

    def run(
        self,
        requirements: FESMRequirementSet,
        *,
        query_input: QueryInput | None = None,
        workspace: ToolWorkspace | None = None,
        trace: ExecutionTrace | None = None,
        executor_feedback: str | None = None,
        model: str | None = None,
    ) -> ExecutorAgentResult:
        """Synchronous wrapper for arun."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    requirements,
                    query_input=query_input,
                    workspace=workspace,
                    trace=trace,
                    executor_feedback=executor_feedback,
                    model=model,
                )
            )
        raise RuntimeError("ExecutorAgent.run cannot be called inside a running event loop")

    def parse_tool_plan(self, parsed_output: dict[str, Any] | str) -> list[ExecutorToolCallPlan]:
        """Parse and validate tool calls from already-produced Executor JSON."""
        if isinstance(parsed_output, str):
            try:
                parsed_output = json.loads(parsed_output)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(f"Executor output is not valid JSON: {exc}") from exc
        rows = parsed_output.get("tool_calls", [])
        if not isinstance(rows, list):
            raise ValidationError("Executor output field 'tool_calls' must be a list")
        if len(rows) > self.config.max_tool_calls:
            raise ValidationError(
                f"Executor produced {len(rows)} tool calls, max is {self.config.max_tool_calls}"
            )
        plans = [
            ExecutorToolCallPlan.from_dict(row, default_step_id=index + 1)
            for index, row in enumerate(rows)
        ]
        if self.config.validate_tool_calls:
            for plan in plans:
                self._validate_tool_call(plan)
        plans.sort(key=lambda item: item.step_id)
        return plans

    async def _execute_tool_plan(
        self,
        tool_plan: list[ExecutorToolCallPlan],
        *,
        workspace: ToolWorkspace,
        trace: ExecutionTrace | None,
        requirements: FESMRequirementSet,
        query_input: QueryInput | None,
    ) -> tuple[list[ToolCallRecord], list[ToolResult]]:
        records: list[ToolCallRecord] = []
        results: list[ToolResult] = []
        grouped: dict[int, list[ExecutorToolCallPlan]] = {}
        for plan in tool_plan:
            grouped.setdefault(plan.step_id, []).append(plan)

        for step_id in sorted(grouped):
            step_plans = grouped[step_id]
            if self.config.parallel_same_step and len(step_plans) > 1:
                step_outputs = await asyncio.gather(
                    *[
                        self._execute_one_tool_call(
                            plan,
                            workspace=workspace,
                            trace=trace,
                            requirements=requirements,
                            query_input=query_input,
                        )
                        for plan in step_plans
                    ]
                )
            else:
                step_outputs = []
                for plan in step_plans:
                    step_outputs.append(
                        await self._execute_one_tool_call(
                            plan,
                            workspace=workspace,
                            trace=trace,
                            requirements=requirements,
                            query_input=query_input,
                        )
                    )
            for record, result in step_outputs:
                records.append(record)
                results.append(result)
                if not result.ok and self.config.stop_on_tool_error:
                    return records, results
        return records, results

    async def _execute_one_tool_call(
        self,
        plan: ExecutorToolCallPlan,
        *,
        workspace: ToolWorkspace,
        trace: ExecutionTrace | None,
        requirements: FESMRequirementSet,
        query_input: QueryInput | None,
    ) -> tuple[ToolCallRecord, ToolResult]:
        request = plan.to_request()
        record = ToolCallRecord(
            tool_name=plan.tool_name,
            arguments=plan.arguments,
            fills=plan.fills,
            call_id=request.call_id,
            metadata={
                "step_id": plan.step_id,
                "output_name": plan.output_name,
                "rationale": plan.rationale,
            },
        )
        record.mark_running()
        if trace is not None:
            trace.add_tool_call(record)

        context = ToolContext(
            workspace=workspace,
            query=requirements.query,
            images=query_input.images if query_input else [],
            metadata={"schema_version": requirements.schema_version},
        )
        result = await self.tool_registry.dispatch(
            request,
            workspace=workspace,
            context=context,
        )
        if result.ok:
            result_refs = {plan.output_name: plan.output_name} if plan.output_name else {}
            record.mark_succeeded(
                result_refs=result_refs,
                result_preview=self._preview_result(result.result),
            )
        else:
            record.mark_failed(result.error or "unknown tool error")
            if trace is not None:
                trace.errors.append(record.error or "unknown tool error")
        if trace is not None:
            trace.workspace_snapshot.update(workspace.raw_values())
        return record, result

    def _validate_tool_call(self, plan: ExecutorToolCallPlan) -> None:
        if not self.tool_registry.contains(plan.tool_name):
            raise ValidationError(f"Executor selected unknown tool: {plan.tool_name}")
        if plan.tool_name == "code" and not self.config.allow_code_tool:
            raise ValidationError("Executor selected code tool while allow_code_tool=False")
        if not plan.output_name and plan.tool_name != "code":
            raise ValidationError(f"Tool call for {plan.tool_name} must include output_name")

    def _available_tool_names(self) -> list[str]:
        names = self.tool_registry.names()
        if self.config.allow_code_tool:
            return names
        return [name for name in names if name != "code"]

    def _text_format(self) -> dict[str, Any] | None:
        if not self.config.use_json_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": "peps_executor_output",
            "schema": EXECUTOR_RESPONSE_JSON_SCHEMA,
            "strict": self.config.strict_json_schema,
        }

    def _preview_result(self, result: Any) -> dict[str, Any]:
        plain = to_plain_data(result)
        text = json.dumps(plain, ensure_ascii=True, default=str)
        if len(text) > 1000:
            return {"preview": text[:1000], "truncated": True}
        if isinstance(plain, dict):
            return plain
        return {"value": plain}

