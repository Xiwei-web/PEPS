"""Static PEPS workflow graph description."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkflowNode:
    name: str
    role: str
    description: str


@dataclass(slots=True)
class WorkflowEdge:
    source: str
    target: str
    condition: str = "always"


@dataclass(slots=True)
class WorkflowGraph:
    """A lightweight graph descriptor for the PEPS agentic workflow."""

    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    edges: list[WorkflowEdge] = field(default_factory=list)

    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, source: str, target: str, *, condition: str = "always") -> None:
        if source not in self.nodes:
            raise ValueError(f"Unknown source node: {source}")
        if target not in self.nodes:
            raise ValueError(f"Unknown target node: {target}")
        self.edges.append(WorkflowEdge(source=source, target=target, condition=condition))

    def next_nodes(self, source: str) -> list[WorkflowEdge]:
        return [edge for edge in self.edges if edge.source == source]

    def as_dict(self) -> dict:
        return {
            "nodes": {
                name: {
                    "role": node.role,
                    "description": node.description,
                }
                for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "condition": edge.condition,
                }
                for edge in self.edges
            ],
        }


def build_default_workflow_graph() -> WorkflowGraph:
    """Return the PEPS Parser -> Executor -> Coder -> Verifier graph."""
    graph = WorkflowGraph()
    for node in [
        WorkflowNode("parser", "agent", "Compile query into a FESM requirement set."),
        WorkflowNode("executor", "agent", "Acquire concrete values for required primitives."),
        WorkflowNode("coder", "agent", "Compute deterministic metrics and answer from workspace."),
        WorkflowNode("verifier", "agent", "Verify answer support and emit typed feedback."),
        WorkflowNode("accept", "terminal", "Accepted answer and trace."),
        WorkflowNode("refine", "control", "Prepare Parser feedback for another round."),
        WorkflowNode("reject", "terminal", "No accepted trace within loop budget."),
    ]:
        graph.add_node(node)
    graph.add_edge("parser", "executor")
    graph.add_edge("executor", "coder")
    graph.add_edge("coder", "verifier")
    graph.add_edge("verifier", "accept", condition="decision=accept")
    graph.add_edge("verifier", "refine", condition="decision=reject and rounds_remaining")
    graph.add_edge("verifier", "reject", condition="decision=reject and no_rounds_remaining")
    graph.add_edge("refine", "parser", condition="missing_slot or primitive_gap feedback")
    return graph

