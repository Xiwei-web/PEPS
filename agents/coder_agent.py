"""Coder Agent for PEPS."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from peps.core.enums import ToolCallStatus, TraceStage
from peps.core.exceptions import LLMOutputError, ValidationError
from peps.core.trace import AgentRunRecord, CodeExecutionRecord, ExecutionTrace
from peps.core.types import FESMRequirementSet, QueryInput
from peps.llm.openai_client import OpenAIClient, OpenAIResponse
from peps.prompts.coder_prompt import (
    CODER_PROMPT_VERSION,
    CODER_RESPONSE_JSON_SCHEMA,
    build_coder_system_prompt,
    build_coder_user_prompt,
)
from peps.tools.code.runtime import CodeRuntime, CodeRuntimeConfig
from peps.tools.code.sandbox import SandboxExecutionResult
from peps.tools.workspace import ToolWorkspace


class CoderLLMClient(Protocol):
    """Minimal client interface required by CoderAgent."""

    def create_response(self, **kwargs: Any) -> OpenAIResponse:
        ...


@dataclass(slots=True)
class CoderAgentConfig:
    """Runtime settings for CoderAgent."""

    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    use_json_schema: bool = True
    strict_json_schema: bool = False
    max_code_attempts: int = 2
    execute_code: bool = True
    runtime: CodeRuntimeConfig | None = None


@dataclass(slots=True)
class CoderAgentResult:
    """Structured result of one Coder invocation."""

    parsed_output: dict[str, Any]
    raw_output: str
    agent_run: AgentRunRecord
    code_record: CodeExecutionRecord
    sandbox_result: SandboxExecutionResult | None = None

    @property
    def answer(self) -> str | None:
        return self.code_record.answer

    @property
    def computed_metrics(self) -> dict[str, Any]:
        return self.code_record.computed_metrics


class CoderAgent:
    """Generate and execute deterministic code for a fixed PEPS trace."""

    def __init__(
        self,
        *,
        llm_client: CoderLLMClient | None = None,
        runtime: CodeRuntime | None = None,
        config: CoderAgentConfig | None = None,
    ) -> None:
        self.config = config or CoderAgentConfig()
        self.llm_client = llm_client or OpenAIClient.from_env()
        self.runtime = runtime or CodeRuntime(self.config.runtime)

    def run(
        self,
        requirements: FESMRequirementSet,
        workspace: ToolWorkspace,
        *,
        query_input: QueryInput | None = None,
        trace: ExecutionTrace | None = None,
        model: str | None = None,
    ) -> CoderAgentResult:
        previous_error: str | None = None
        last_result: CoderAgentResult | None = None
        attempts = max(1, self.config.max_code_attempts)

        for attempt in range(attempts):
            result = self._run_once(
                requirements,
                workspace,
                query_input=query_input,
                trace=trace,
                model=model,
                previous_error=previous_error,
                attempt_index=attempt,
            )
            last_result = result
            if not self.config.execute_code:
                return result
            if result.sandbox_result is not None and result.sandbox_result.ok:
                return result
            previous_error = (
                result.sandbox_result.error
                if result.sandbox_result is not None
                else result.code_record.error
            )
        assert last_result is not None
        return last_result

    def _run_once(
        self,
        requirements: FESMRequirementSet,
        workspace: ToolWorkspace,
        *,
        query_input: QueryInput | None,
        trace: ExecutionTrace | None,
        model: str | None,
        previous_error: str | None,
        attempt_index: int,
    ) -> CoderAgentResult:
        system_prompt = build_coder_system_prompt()
        user_prompt = build_coder_user_prompt(
            requirements,
            workspace,
            query_input=query_input,
            previous_error=previous_error,
        )
        agent_run = AgentRunRecord(
            agent_name="CoderAgent",
            stage=TraceStage.CODER,
            prompt_name=CODER_PROMPT_VERSION,
            status=ToolCallStatus.RUNNING,
            input_summary={
                "query_id": query_input.query_id if query_input else None,
                "schema_version": requirements.schema_version,
                "workspace_names": workspace.names(),
                "attempt_index": attempt_index,
            },
        )
        try:
            response = self.llm_client.create_response(
                system_prompt=system_prompt,
                user_text=user_prompt,
                model=model or self.config.model,
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                text_format=self._text_format(),
            )
            parsed_output = response.json_object()
            code = self._extract_code(parsed_output)
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

        code_record = CodeExecutionRecord(
            code=code,
            inputs={"workspace_keys": workspace.names()},
            metadata={"attempt_index": attempt_index},
        )
        sandbox_result: SandboxExecutionResult | None = None
        if self.config.execute_code:
            code_record.mark_running()
            sandbox_result = self.runtime.execute(code, workspace)
            if sandbox_result.ok:
                code_record.mark_succeeded(
                    outputs=sandbox_result.outputs,
                    answer=sandbox_result.answer,
                    computed_metrics=sandbox_result.computed_metrics,
                )
            else:
                code_record.mark_failed(sandbox_result.error or "sandbox execution failed")
                if trace is not None:
                    trace.errors.append(code_record.error or "sandbox execution failed")
        if trace is not None:
            trace.add_code_run(code_record)
        return CoderAgentResult(
            parsed_output=parsed_output,
            raw_output=response.text,
            agent_run=agent_run,
            code_record=code_record,
            sandbox_result=sandbox_result,
        )

    def parse_output(self, output: dict[str, Any] | str) -> str:
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(f"Coder output is not valid JSON: {exc}") from exc
        return self._extract_code(output)

    def _extract_code(self, parsed_output: dict[str, Any]) -> str:
        code = parsed_output.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValidationError("Coder output must include non-empty code string")
        code = code.strip()
        if code.startswith("```"):
            raise ValidationError("Coder code string must not include markdown fences")
        if "def execute" not in code:
            raise ValidationError("Coder code must define execute(workspace)")
        return code

    def _text_format(self) -> dict[str, Any] | None:
        if not self.config.use_json_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "name": "peps_coder_output",
            "schema": CODER_RESPONSE_JSON_SCHEMA,
            "strict": self.config.strict_json_schema,
        }

