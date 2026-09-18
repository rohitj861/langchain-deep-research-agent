# Deep Research Agent (LangChain Deep Agents + Streamlit)

Ask a research question, pick a depth, and a team of agents researches the web, writes a report, and critiques it.
You can read the result in the app or download it as **Markdown** or **PDF**.

## How it works

```
User question + depth
        │
        ▼
 Orchestrator (create_deep_agent)
   1. task → research-agent      Tavily web search, returns findings + URLs
   2. task → synthesizer-agent   writes /final_report.md (with comparison table and references)
   3. task → critique-agent      reviews and tightens /final_report.md (1-2 passes)
        │
        ▼
 Streamlit: rendered report · .md download · .pdf download
```

| Depth    | Searches | Tavily results / depth | Report length     | Critique passes |
|----------|----------|------------------------|-------------------|-----------------|
| Basic    | 2-3      | 3 / basic              | ~500-800 words    | 1 |
| Standard | 4-6      | 5 / advanced           | ~1,200-1,800 words| 1 |
| Advanced | 8-12     | 8 / advanced           | ~2,500-3,500 words| 2 |

You can change the presets in `research_agent/config.py`.

**Tavily-only sourcing, top 5 references**
- The only search tool is Tavily (`research_agent/sources.py`). deepagents' built-in `general-purpose` subagent
  is replaced with a Tavily-only version, and the prompts forbid answering from the model's own knowledge.
- Every Tavily result is recorded in a `SourceRegistry`. The synthesizer and critique agents pick references from
  it with the `list_tavily_sources` tool.
- After the run, `enforce_references()` removes any URL that Tavily did not return, keeps at most **5** references
  (filling up from the highest-ranked Tavily results if fewer are cited), and renumbers the inline `[n]` citations to match.

**Live UI**: a Research → Synthesize → Critique tracker, live Tavily search and source counters, a list of sources
ranked by relevance, and a references tab showing each reference's Tavily score and the query that found it.

**Human in the loop** (sidebar toggle): the agent pauses before each subagent runs. You then approve or reject that
step, and you can give feedback when you reject. This uses deepagents `interrupt_on={"task": ...}` together with a checkpointer.

## Project layout

```
app.py                         Streamlit UI
research_agent/
  config.py                    depth presets, model name
  agent.py                     orchestrator + 3 subagents (create_deep_agent)
  runner.py                    streaming progress, interrupts/resume, report extraction
  sources.py                   Tavily-only search tool, source registry, 5-reference enforcement
  pdf_export.py                Markdown → PDF (fpdf2, Unicode font support)
tests/                         offline tests (scripted fake model, reference enforcement)
requirements.txt / packages.txt
.env.example / .streamlit/secrets.toml.example
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then fill in OPENAI_API_KEY and TAVILY_API_KEY
streamlit run app.py
```

Get keys from https://platform.openai.com/api-keys and https://app.tavily.com.
To use a different model, set `DEEP_AGENT_MODEL` (for example `anthropic:claude-sonnet-5`, after `pip install langchain-anthropic`)
or change it in the sidebar.

Tests (these run offline):

```bash
pip install pytest
python -m pytest tests -q
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` already keeps `.env` and `secrets.toml` out of git.
2. Go to https://share.streamlit.io → **Create app** and pick the repo, branch, and `app.py`.
3. Under **Advanced settings**, choose Python **3.12** and paste your secrets:
   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   DEEP_AGENT_MODEL = "openai:gpt-5.4-mini"
   ```
4. Deploy. `packages.txt` installs the DejaVu fonts, so PDFs render Unicode text correctly.

## Notes

- If you deploy publicly without secrets, each visitor can enter their own keys in the sidebar. Those keys are
  passed straight to that visitor's agent and never shared with other sessions. Keys typed in the sidebar only apply
  to `openai:` models; other providers read their key from secrets/env.
- **User-initiated runs only.** The question box always starts empty, and the question comes only from the user.
  A run starts only when the user clicks **Run research**: Enter and Ctrl+Enter don't submit, and page reruns never start
  or restart a run. If a run is interrupted (for example by clicking during research), the app pauses and asks whether
  to **Continue this run** (from its last checkpoint) or **Discard** it.
- Agent state is held in memory (`InMemorySaver`), so a browser refresh starts over.
