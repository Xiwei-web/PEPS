"""Subprocess sandbox for deterministic PEPS code execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import subprocess
import sys
import time
from typing import Any

from peps.core.io import to_plain_data


SANDBOX_RUNTIME = r'''
import ast
import json
import math
import statistics
import itertools
import functools
import collections
import operator
import traceback


FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "input", "__import__", "globals",
    "locals", "vars", "dir", "help", "setattr", "delattr", "breakpoint",
}
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests",
    "urllib", "http", "ftplib", "pickle", "cloudpickle", "multiprocessing",
    "threading", "asyncio", "importlib",
}
ALLOWED_IMPORTS = {
    "math", "statistics", "itertools", "functools", "collections", "operator",
}


def make_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(v) for v in value]
    return repr(value)


def validate_ast(code):
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Delete, ast.Global, ast.Nonlocal)):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.Raise)):
            raise ValueError(f"Forbidden control construct: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS or root in FORBIDDEN_MODULES:
                    raise ValueError(f"Forbidden import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root not in ALLOWED_IMPORTS or root in FORBIDDEN_MODULES:
                raise ValueError(f"Forbidden import: {module}")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError(f"Forbidden attribute: {node.attr}")
    return tree


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS or root in FORBIDDEN_MODULES:
        raise ImportError(f"Import not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": safe_import,
}


def run():
    payload = json.loads(__import__("sys").stdin.read())
    code = payload["code"]
    workspace = payload.get("workspace", {})
    tree = validate_ast(code)
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
        "operator": operator,
    }
    exec(compile(tree, "<peps_generated_code>", "exec"), namespace, namespace)
    execute = namespace.get("execute")
    if not callable(execute):
        raise ValueError("Generated code must define callable execute(workspace)")
    result = execute(workspace)
    if not isinstance(result, dict):
        result = {"answer": str(result), "computed_metrics": {"result": result}, "raw_result": result}
    print(json.dumps({"ok": True, "result": make_jsonable(result)}, ensure_ascii=True))


try:
    run()
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }, ensure_ascii=True))
'''


@dataclass(slots=True)
class SandboxConfig:
    """Runtime limits for generated code execution."""

    timeout_seconds: float = 10.0
    python_executable: str = sys.executable
    isolated: bool = True


@dataclass(slots=True)
class SandboxExecutionResult:
    """Result from running generated code in the sandbox."""

    ok: bool
    code: str
    outputs: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    computed_metrics: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    traceback: str | None = None
    runtime_seconds: float = 0.0


class CodeSandbox:
    """Execute generated PEPS code in a constrained subprocess."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def run(self, code: str, workspace: dict[str, Any]) -> SandboxExecutionResult:
        start = time.monotonic()
        payload = json.dumps(
            {"code": code, "workspace": to_plain_data(workspace)},
            ensure_ascii=True,
        )
        cmd = [self.config.python_executable]
        if self.config.isolated:
            cmd.append("-I")
        cmd.extend(["-c", SANDBOX_RUNTIME])

        try:
            completed = subprocess.run(
                cmd,
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxExecutionResult(
                ok=False,
                code=code,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error=f"Code execution timed out after {self.config.timeout_seconds} seconds.",
                runtime_seconds=time.monotonic() - start,
            )

        runtime_seconds = time.monotonic() - start
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            return SandboxExecutionResult(
                ok=False,
                code=code,
                stdout=stdout,
                stderr=stderr,
                error=f"Sandbox subprocess exited with code {completed.returncode}.",
                runtime_seconds=runtime_seconds,
            )
        if not stdout:
            return SandboxExecutionResult(
                ok=False,
                code=code,
                stdout=stdout,
                stderr=stderr,
                error="Sandbox produced no stdout.",
                runtime_seconds=runtime_seconds,
            )
        try:
            payload_out = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return SandboxExecutionResult(
                ok=False,
                code=code,
                stdout=stdout,
                stderr=stderr,
                error=f"Sandbox returned invalid JSON: {exc}",
                runtime_seconds=runtime_seconds,
            )
        if not payload_out.get("ok"):
            return SandboxExecutionResult(
                ok=False,
                code=code,
                stdout=stdout,
                stderr=stderr,
                error=payload_out.get("error", "unknown sandbox error"),
                traceback=payload_out.get("traceback"),
                runtime_seconds=runtime_seconds,
            )
        outputs = payload_out.get("result", {})
        return SandboxExecutionResult(
            ok=True,
            code=code,
            outputs=outputs,
            answer=outputs.get("answer"),
            computed_metrics=outputs.get("computed_metrics", {}),
            stdout=stdout,
            stderr=stderr,
            runtime_seconds=runtime_seconds,
        )

