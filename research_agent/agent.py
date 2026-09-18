"""Deep Agent: orchestrator + research / synthesizer / critique subagents."""

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_tavily._utilities import TavilySearchAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver

from .config import DEPTH_PRESETS, REPORT_PATH, get_model_name
from .sources import MAX_REFERENCES, SourceRegistry, make_tavily_tools

ORCHESTRATOR_PROMPT = """You are the orchestrator of a research team. Follow these steps strictly and in order.
Every step MUST be delegated to a subagent with the `task` tool. Do not research or write the report yourself.

1. Call `research-agent` with the user's question. Ask it to run {num_queries} Tavily searches and return
   detailed findings with source URLs. All information must come from Tavily search; never answer from your own knowledge.
2. Call `synthesizer-agent`. Pass it the user's question AND the full findings from step 1 verbatim.
   It must write the report to `{report_path}`.
3. Call `critique-agent` {critique_passes} time(s). It reviews and tightens `{report_path}` in place.

When all steps are done, reply with a 2-3 sentence summary of the report. Do not paste the full report.
"""

RESEARCH_PROMPT = """You are an expert research agent. Your ONLY source of information is the `tavily_search` tool
(the Tavily search API). Do not use your own background knowledge: every fact you report must come from a Tavily result.

- Run {num_queries} distinct, well-targeted searches covering different angles of the question
  (definitions, current state, data/statistics, comparisons, risks, expert opinions, recent developments).
- Prefer authoritative, recent sources. Note publication dates when available.
- Return ONLY your findings as structured notes: one section per sub-topic, each fact followed by its
  source as [Title](URL). End with a deduplicated list of all sources.
- Only cite URLs that appeared in a Tavily result. Do not write the final report. Do not invent facts or URLs.
"""

SYNTHESIZER_PROMPT = """You are a synthesizer agent. Turn research findings into a polished Markdown report
and save it with `write_file` to `{report_path}`.

Report structure:
# <Descriptive title>
## Executive Summary
## Key Findings  (several subsections as needed)
## Comparison Table  (a Markdown table comparing the main options/approaches/entities in the findings)
## Conclusion
## References  (numbered list of [Title](URL))

References rules (strict):
- First call `list_tavily_sources` to get the sources Tavily returned. Choose EXACTLY {max_refs} references from it
  (fewer only if Tavily returned fewer): the most important, authoritative, and most-used sources.
- Every reference URL must appear in `list_tavily_sources`, copied exactly. No other URLs anywhere in the report.
- Cite inline only as [1]..[{max_refs}] matching the References list.

Target length: {report_length}. Use only information from the findings you were given (all from Tavily).
After writing the file, reply "Report written."
"""

CRITIQUE_PROMPT = """You are a critique agent. Read `{report_path}` with `read_file`, then review it for:
accuracy against its cited sources, unsupported claims, redundancy, unclear wording, weak structure,
a missing or poor comparison table, and broken or missing references.
Call `list_tavily_sources` and verify that the References list has at most {max_refs} entries, every URL appears in
that list, and every inline citation is in [1]..[{max_refs}]. Remove any claim that relies on a non-Tavily source.

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
        "max_refs": MAX_REFERENCES,
    }

    registry = SourceRegistry()
    tavily_search, list_tavily_sources = make_tavily_tools(
        registry,
        max_results=preset.max_results,
        search_depth=preset.search_depth,
        api_wrapper=TavilySearchAPIWrapper(tavily_api_key=tavily_api_key) if tavily_api_key else None,
    )

    model = model or get_model_name()
    if isinstance(model, str) and openai_api_key and model.startswith("openai:"):
        model = init_chat_model(model, api_key=openai_api_key)

    research_prompt = RESEARCH_PROMPT.format(**fmt)
    subagents = [
        # Override deepagents' default general-purpose subagent so no path can research without Tavily.
        {
            "name": "general-purpose",
            "description": "Tavily-only research helper. Same rules as research-agent.",
            "system_prompt": research_prompt,
            "tools": [tavily_search],
        },
        {
            "name": "research-agent",
            "description": "Runs web searches with Tavily and returns detailed findings with source URLs.",
            "system_prompt": research_prompt,
            "tools": [tavily_search],
        },
        {
            "name": "synthesizer-agent",
            "description": f"Synthesizes research findings into {REPORT_PATH} with a comparison table and references.",
            "system_prompt": SYNTHESIZER_PROMPT.format(**fmt),
            "tools": [list_tavily_sources],
        },
        {
            "name": "critique-agent",
            "description": f"Reviews and tightens {REPORT_PATH} in place.",
            "system_prompt": CRITIQUE_PROMPT.format(**fmt),
            "tools": [list_tavily_sources],
        },
    ]

    # Human in the loop: pause before each subagent is launched so the user can approve or reject it.
    interrupt_on = {"task": {"allowed_decisions": ["approve", "reject"]}} if human_in_the_loop else None

    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=ORCHESTRATOR_PROMPT.format(**fmt),
        subagents=subagents,
        interrupt_on=interrupt_on,
        checkpointer=InMemorySaver(),  # required for interrupts / resume
    )
    agent.source_registry = registry  # read by the runner/UI for live sources and reference enforcement
    return agent
