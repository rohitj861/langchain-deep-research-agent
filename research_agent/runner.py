"""Drive the agent: stream progress events, surface interrupts, extract the final report."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.utils import file_data_to_string
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from .config import REPORT_PATH


@dataclass
class ProgressEvent:
    source: str   # "orchestrator" or the subagent namespace
    kind: str     # "tool_call" | "tool_result" | "message"
    text: str


@dataclass
class RunOutcome:
    status: str                                  # "done" | "interrupted"
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    report: str | None = None
    summary: str | None = None


def _shorten(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _describe_tool_call(call: dict) -> str:
    name, args = call["name"], call.get("args", {})
    if name == "task":
        return f"Launching **{args.get('subagent_type', '?')}**: {_shorten(args.get('description', ''), 200)}"
    if name == "tavily_search":
        return f"Searching the web: _{args.get('query', '')}_"
    if name in ("write_file", "edit_file", "read_file"):
        return f"`{name}` → {args.get('file_path', '')}"
    return f"`{name}` {_shorten(args, 150)}"


def stream_run(agent, payload, thread_id: str) -> Iterator[ProgressEvent | RunOutcome]:
    """Run (or resume) the agent, yielding ProgressEvents and finally a RunOutcome.

    `payload` is either the initial input dict or a `Command(resume=...)`.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    interrupts: list = []

    for namespace, chunk in agent.stream(payload, config=config, stream_mode="updates", subgraphs=True):
        source = "orchestrator" if not namespace else namespace[0].split(":")[0]
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts.extend(update)
                continue
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []) or []:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        yield ProgressEvent(source, "tool_call", _describe_tool_call(call))
                    if not msg.tool_calls and msg.content:
                        yield ProgressEvent(source, "message", _shorten(msg.text, 400))
                elif isinstance(msg, ToolMessage) and msg.name != "task":
                    yield ProgressEvent(source, "tool_result", f"`{msg.name}` returned {len(str(msg.content)):,} chars")

    if interrupts:
        actions = []
        for intr in interrupts:
            value = getattr(intr, "value", intr) or {}
            actions.extend(value.get("action_requests", []))
        yield RunOutcome(status="interrupted", pending_actions=actions)
        return

    state = agent.get_state(config).values
    files = state.get("files", {}) or {}
    report = file_data_to_string(files[REPORT_PATH]) if REPORT_PATH in files else None
    summary = next((m.text for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage) and m.text), None)
    yield RunOutcome(status="done", report=report, summary=summary)


def initial_input(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def resume_command(decisions: list[str], reason: str | None = None) -> Command:
    """Build a resume Command from a list of "approve"/"reject" decisions."""
    out = []
    for d in decisions:
        if d == "reject":
            out.append({"type": "reject", "message": reason or "The user rejected this step."})
        else:
            out.append({"type": "approve"})
    return Command(resume={"decisions": out})
