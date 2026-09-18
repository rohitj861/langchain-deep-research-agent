"""Offline smoke test: runs the full orchestrator -> subagent pipeline with a scripted fake model.

Run with:  python -m pytest tests -q   (no API keys or network needed)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from research_agent import build_agent, initial_input, markdown_to_pdf, resume_command, stream_run
from research_agent.runner import RunOutcome

REPORT = """# Vector DBs Compared

## Executive Summary
Short summary with “smart quotes” — and a dash [1].

## Comparison Table
| DB | Hosting | Notes |
|----|---------|-------|
| Alpha | Managed | Fast ✓ |
| Beta | Self-hosted | Cheap |

## References
1. [Example](https://example.com)
"""


class ScriptedModel(BaseChatModel):
    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def make_script():
    return [
        _call("task", {"subagent_type": "research-agent", "description": "Research vector DBs"}, 1),
        AIMessage(content="Findings: Alpha is fast [Example](https://example.com)."),
        _call("task", {"subagent_type": "synthesizer-agent", "description": "Write the report"}, 2),
        _call("write_file", {"file_path": "/final_report.md", "content": REPORT}, 3),
        AIMessage(content="Report written."),
        _call("task", {"subagent_type": "critique-agent", "description": "Review the report"}, 4),
        AIMessage(content="- No changes needed."),
        AIMessage(content="The report compares Alpha and Beta."),
    ]


def _run_to_end(agent, thread, approve=False):
    payload = initial_input("Compare vector DBs")
    interrupts = 0
    while True:
        outcome = [x for x in stream_run(agent, payload, thread) if isinstance(x, RunOutcome)][-1]
        if outcome.status == "done":
            return outcome, interrupts
        assert approve, "unexpected interrupt"
        interrupts += 1
        payload = resume_command(["approve"] * len(outcome.pending_actions))


def test_pipeline_produces_report():
    agent = build_agent("basic", model=ScriptedModel(script=make_script()))
    outcome, interrupts = _run_to_end(agent, "t1")
    assert interrupts == 0
    assert outcome.report and "Comparison Table" in outcome.report
    assert outcome.summary == "The report compares Alpha and Beta."


def test_human_in_the_loop_pauses_before_each_subagent():
    agent = build_agent("standard", model=ScriptedModel(script=make_script()), human_in_the_loop=True)
    outcome, interrupts = _run_to_end(agent, "t2", approve=True)
    assert interrupts == 3
    assert outcome.report


def test_pdf_export():
    pdf = markdown_to_pdf(REPORT, title="t")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
