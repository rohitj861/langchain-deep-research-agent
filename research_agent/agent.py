"""Deep Agent: orchestrator + research / synthesizer / critique subagents."""

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver

from .config import DEPTH_PRESETS, REPORT_PATH, get_model_name

ORCHESTRATOR_PROMPT = """You are the orchestrator of a research team. Follow these steps strictly and in order.
Every step MUST be delegated to a subagent with the `task` tool. Do not research or write the report yourself.

1. Call `research-agent` with the user's question. Ask it to run {num_queries} web searches and return
   detailed findings with source URLs.
2. Call `synthesizer-agent`. Pass it the user's question AND the full findings from step 1 verbatim.
   It must write the report to `{report_path}`.
3. Call `critique-agent` {critique_passes} time(s). It reviews and tightens `{report_path}` in place.

When all steps are done, reply with a 2-3 sentence summary of the report. Do not paste the full report.
"""

RESEARCH_PROMPT = """You are an expert research agent. Use the `tavily_search` tool to answer the research question.

- Run {num_queries} distinct, well-targeted searches covering different angles of the question
  (definitions, current state, data/statistics, comparisons, risks, expert opinions, recent developments).
- Prefer authoritative, recent sources. Note publication dates when available.
- Return ONLY your findings as structured notes: one section per sub-topic, each fact followed by its
  source as [Title](URL). End with a deduplicated list of all sources.
- Do not write the final report. Do not invent facts or URLs.
"""

SYNTHESIZER_PROMPT = """You are a synthesizer agent. Turn research findings into a polished Markdown report
and save it with `write_file` to `{report_path}`.

Report structure:
# <Descriptive title>
## Executive Summary
## Key Findings  (several subsections as needed)
## Comparison Table  (a Markdown table comparing the main options/approaches/entities in the findings)
## Conclusion
## References  (numbered list of [Title](URL), only sources present in the findings)

Target length: {report_length}. Cite sources inline as [n] matching the References list.
Use only information from the findings you were given. After writing the file, reply "Report written."
"""

CRITIQUE_PROMPT = """You are a critique agent. Read `{report_path}` with `read_file`, then review it for:
accuracy against its cited sources, unsupported claims, redundancy, unclear wording, weak structure,
a missing or poor comparison table, and broken or missing references.

Fix the problems directly by rewriting the file with `write_file` (or `edit_file` for small changes).
Keep all correct citations. Keep it tight: cut filler, not substance.
Reply with a short bullet list of what you changed.
"""


def build_agent(
    depth: str = "standard",
    model=None,
    human_in_the_loop: bool = False,
    openai_api_key: str | None = None,
    tavily_api_key: str | None = None,
):
    """Create the research deep agent for the given depth preset.

    API keys passed here are scoped to this agent; when omitted, the clients read them from env vars.
    """
    preset = DEPTH_PRESETS[depth]
    fmt = {
        "num_queries": preset.num_queries,
        "report_length": preset.report_length,
        "critique_passes": preset.critique_passes,
        "report_path": REPORT_PATH,
    }

    tavily_kwargs = {"api_wrapper": TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)} if tavily_api_key else {}
    tavily_search = TavilySearch(max_results=preset.max_results, search_depth=preset.search_depth, **tavily_kwargs)

    model = model or get_model_name()
    if isinstance(model, str) and openai_api_key and model.startswith("openai:"):
        model = init_chat_model(model, api_key=openai_api_key)

    subagents = [
        {
            "name": "research-agent",
            "description": "Runs web searches with Tavily and returns detailed findings with source URLs.",
            "system_prompt": RESEARCH_PROMPT.format(**fmt),
            "tools": [tavily_search],
        },
        {
            "name": "synthesizer-agent",
            "description": f"Synthesizes research findings into {REPORT_PATH} with a comparison table and references.",
            "system_prompt": SYNTHESIZER_PROMPT.format(**fmt),
            "tools": [],
        },
        {
            "name": "critique-agent",
            "description": f"Reviews and tightens {REPORT_PATH} in place.",
            "system_prompt": CRITIQUE_PROMPT.format(**fmt),
            "tools": [],
        },
    ]

    # Human in the loop: pause before each subagent is launched so the user can approve or reject it.
    interrupt_on = {"task": {"allowed_decisions": ["approve", "reject"]}} if human_in_the_loop else None

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=ORCHESTRATOR_PROMPT.format(**fmt),
        subagents=subagents,
        interrupt_on=interrupt_on,
        checkpointer=InMemorySaver(),  # required for interrupts / resume
    )
