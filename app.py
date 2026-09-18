"""Streamlit front end for the LangChain Deep Agents research assistant."""

import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

KEY_NAMES = ("OPENAI_API_KEY", "TAVILY_API_KEY")


def _load_secrets() -> None:
    """Copy Streamlit Cloud secrets into env vars so LangChain clients pick them up."""
    try:
        for k in (*KEY_NAMES, "DEEP_AGENT_MODEL"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except FileNotFoundError:
        pass
    except Exception:
        pass


_load_secrets()

# Treat unfilled placeholders from .env / secrets.toml as missing keys.
for _k in KEY_NAMES:
    if os.getenv(_k, "").startswith("your-") or os.getenv(_k, "").endswith("..."):
        del os.environ[_k]

from research_agent import (  # noqa: E402  (import after env vars are set)
    DEPTH_PRESETS,
    ProgressEvent,
    RunOutcome,
    build_agent,
    initial_input,
    markdown_to_pdf,
    resume_command,
    stream_run,
)
from research_agent.config import get_model_name  # noqa: E402

# ---------------------------------------------------------------- session state
defaults = {
    "phase": "idle",          # idle | running | awaiting_approval | done | error
    "agent": None,
    "thread_id": None,
    "question": "",
    "depth": "standard",
    "events": [],
    "pending": [],
    "resume_payload": None,
    "report": None,
    "summary": None,
    "error": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)
S = st.session_state


def reset():
    for k, v in defaults.items():
        S[k] = v


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    # Keys typed here stay in this browser session only (never written to os.environ,
    # which is shared by all visitors of a deployed app).
    session_keys: dict[str, str] = {}
    for key in KEY_NAMES:
        if not os.getenv(key):
            val = st.text_input(key, type="password", help="Not found in secrets/env. Used for this session only.")
            if val:
                session_keys[key] = val
        else:
            st.success(f"{key} loaded", icon="🔑")

    model = st.text_input("Model", value=get_model_name(), help="Any LangChain `provider:model` string.")
    hitl = st.toggle(
        "Human in the loop",
        value=False,
        help="Pause before each subagent (research, synthesizer, critique) so you can approve or reject it.",
    )
    st.divider()
    st.caption(
        "**Pipeline**\n\n"
        "1. 🔍 research-agent: Tavily web search\n"
        "2. ✍️ synthesizer-agent: writes `final_report.md`\n"
        "3. 🧐 critique-agent: reviews & tightens"
    )
    if S.phase != "idle" and st.button("Start new research", use_container_width=True):
        reset()
        st.rerun()

# ---------------------------------------------------------------- header / input
st.title("🔎 Deep Research Agent")
st.caption("LangChain Deep Agents · orchestrator + research / synthesizer / critique subagents")

locked = S.phase in ("running", "awaiting_approval")

question = st.text_area(
    "What do you want to research?",
    value=S.question,
    placeholder="e.g. Compare the leading vector databases for production RAG in 2026",
    height=100,
    disabled=locked,
)

depth = st.radio(
    "Research depth",
    options=list(DEPTH_PRESETS),
    format_func=lambda k: f"{DEPTH_PRESETS[k].label}: {DEPTH_PRESETS[k].description}",
    index=list(DEPTH_PRESETS).index(S.depth),
    disabled=locked,
)

if st.button("🚀 Run research", type="primary", disabled=locked or not question.strip()):
    missing = [k for k in KEY_NAMES if not (os.getenv(k) or session_keys.get(k))]
    if missing:
        st.error(f"Missing API key(s): {', '.join(missing)}. Add them in the sidebar or in secrets.")
    else:
        reset()
        S.question, S.depth = question.strip(), depth
        S.agent = build_agent(
            depth=depth,
            model=model,
            human_in_the_loop=hitl,
            openai_api_key=session_keys.get("OPENAI_API_KEY"),
            tavily_api_key=session_keys.get("TAVILY_API_KEY"),
        )
        S.thread_id = str(uuid.uuid4())
        S.resume_payload = initial_input(S.question)
        S.phase = "running"
        st.rerun()


# ---------------------------------------------------------------- progress log
ICONS = {"tool_call": "▶️", "tool_result": "✅", "message": "💬"}


def render_event(ev: ProgressEvent, container) -> None:
    container.markdown(f"{ICONS.get(ev.kind, '•')} `{ev.source}` {ev.text}")


def show_log(expanded: bool) -> None:
    if S.events:
        with st.expander(f"Agent activity ({len(S.events)} steps)", expanded=expanded):
            for ev in S.events:
                render_event(ev, st)


# ---------------------------------------------------------------- running
if S.phase == "running":
    show_log(expanded=False)
    with st.status(f"Researching ({DEPTH_PRESETS[S.depth].label})...", expanded=True) as status:
        try:
            for item in stream_run(S.agent, S.resume_payload, S.thread_id):
                if isinstance(item, ProgressEvent):
                    S.events.append(item)
                    render_event(item, st)
                elif isinstance(item, RunOutcome):
                    if item.status == "interrupted":
                        S.pending = item.pending_actions
                        S.phase = "awaiting_approval"
                        status.update(label="Waiting for your approval", state="running")
                    else:
                        S.report, S.summary = item.report, item.summary
                        S.phase = "done"
                        status.update(label="Research complete", state="complete")
        except Exception as e:  # surface API/auth errors in the UI
            S.error = f"{type(e).__name__}: {e}"
            S.phase = "error"
            status.update(label="Run failed", state="error")
    st.rerun()

# ---------------------------------------------------------------- human in the loop
if S.phase == "awaiting_approval":
    show_log(expanded=False)
    st.subheader("✋ Approval needed")
    decisions = []
    for i, action in enumerate(S.pending):
        args = action.get("args", {})
        with st.container(border=True):
            st.markdown(f"**Step:** `{args.get('subagent_type', action.get('name'))}`")
            st.markdown(args.get("description", "") or str(args))
            decisions.append(
                st.radio("Decision", ["approve", "reject"], key=f"decision_{len(S.events)}_{i}", horizontal=True)
            )
    reason = st.text_input("Feedback for the agent (used if you reject)", key=f"reason_{len(S.events)}")
    if st.button("Submit decision", type="primary"):
        S.resume_payload = resume_command(decisions, reason or None)
        S.pending = []
        S.phase = "running"
        st.rerun()

# ---------------------------------------------------------------- error
if S.phase == "error":
    show_log(expanded=True)
    st.error(S.error)

# ---------------------------------------------------------------- results
if S.phase == "done":
    show_log(expanded=False)
    if not S.report:
        st.warning("The agent finished but no `final_report.md` was produced.")
        if S.summary:
            st.markdown(S.summary)
    else:
        st.success("Your report is ready. Read it below or download it as Markdown or PDF.")
        slug = re.sub(r"[^a-z0-9]+", "_", S.question.lower()).strip("_")[:50] or "report"

        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download Markdown (.md)",
            data=S.report.encode("utf-8"),
            file_name=f"{slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        try:
            pdf_bytes = markdown_to_pdf(S.report, title=S.question)
            c2.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            c2.error(f"PDF export failed: {e}")

        tab_view, tab_raw = st.tabs(["📄 Rendered report", "📝 Markdown source"])
        with tab_view:
            st.markdown(S.report)
        with tab_raw:
            st.code(S.report, language="markdown")
