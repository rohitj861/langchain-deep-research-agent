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

**Human in the loop** (sidebar toggle): the agent pauses before each subagent runs. You then approve or reject that
step, and you can give feedback when you reject. This uses deepagents `interrupt_on={"task": ...}` together with a checkpointer.

## Project layout

```
app.py                         Streamlit UI
research_agent/
  config.py                    depth presets, model name
  agent.py                     orchestrator + 3 subagents (create_deep_agent)
  runner.py                    streaming progress, interrupts/resume, report extraction
  pdf_export.py                Markdown → PDF (fpdf2, Unicode font support)
tests/test_pipeline.py         offline test with a scripted fake model (no API keys needed)
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
- Agent state is held in memory (`InMemorySaver`), so a browser refresh starts over. Avoid interacting with the page
  while a run is in progress, because Streamlit reruns will interrupt the stream.
