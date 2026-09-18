"""Research depth presets and runtime settings."""

import os
from dataclasses import dataclass

DEFAULT_MODEL = "openai:gpt-5.4-mini"
REPORT_PATH = "/final_report.md"


@dataclass(frozen=True)
class DepthPreset:
    label: str
    description: str
    num_queries: str          # how many distinct searches the researcher should run
    max_results: int          # Tavily results per search
    search_depth: str         # Tavily "basic" | "advanced"
    report_length: str        # target length for the synthesizer
    critique_passes: int      # how many review/tighten passes


DEPTH_PRESETS: dict[str, DepthPreset] = {
    "basic": DepthPreset(
        label="Basic",
        description="Quick overview: 2-3 searches, short report.",
        num_queries="2-3",
        max_results=3,
        search_depth="basic",
        report_length="about 500-800 words",
        critique_passes=1,
    ),
    "standard": DepthPreset(
        label="Standard",
        description="Balanced: 4-6 searches, multi-section report.",
        num_queries="4-6",
        max_results=5,
        search_depth="advanced",
        report_length="about 1,200-1,800 words",
        critique_passes=1,
    ),
    "advanced": DepthPreset(
        label="Advanced",
        description="Deep dive: 8-12 searches across sub-topics, thorough report.",
        num_queries="8-12",
        max_results=8,
        search_depth="advanced",
        report_length="about 2,500-3,500 words",
        critique_passes=2,
    ),
}


def get_model_name() -> str:
    return os.getenv("DEEP_AGENT_MODEL", DEFAULT_MODEL)
