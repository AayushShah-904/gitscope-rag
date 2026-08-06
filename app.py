import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 1. Configure the page's appearance and title
st.set_page_config(
    page_title="GitScope RAG - Codebase Analyzer",
    layout="wide",
)

st.title("GitScope RAG")
st.write("Analyze, search, and understand complex GitHub repositories with advanced context engineering")
st.info(
    "☁️ **Cloud Notice:** This app runs on a free server with ephemeral storage. "
    "Indexed repositories are cleared on each server restart — simply re-enter the URL to re-index.",
    icon="☁️"
)

# 2. Initialize session state variables
if "repo_url"    not in st.session_state: st.session_state.repo_url    = ""
if "repo_id"     not in st.session_state: st.session_state.repo_id     = None
if "repo_dir"    not in st.session_state: st.session_state.repo_dir    = ""   # path to cloned repo on disk
if "qa"          not in st.session_state: st.session_state.qa          = None
if "messages"    not in st.session_state: st.session_state.messages    = []
if "summarized"  not in st.session_state: st.session_state.summarized  = False
if "arch_report" not in st.session_state: st.session_state.arch_report = ""   # cached architecture report


# 3. Create the sidebar where users can input the GitHub repository URL
with st.sidebar:
    st.header("Repository Configuration")

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        value=st.session_state.repo_url,
    )

    force = st.checkbox("Force re-ingest", value=False,
                        help="Force re-cloning and embedding even if already indexed in ChromaDB")

    ingest_btn = st.button("Load & Index Repository", type="primary", use_container_width=True)

    st.divider()
    
    st.markdown("### Chat Management")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summarized = False
        if st.session_state.qa:
            st.session_state.qa.clear_history()
        logger.info("Chat history cleared by user.")
        st.success("Chat history cleared!")
        st.rerun()

    # ── Architecture Report section (only shown when a repo is loaded) ──
    if st.session_state.qa:
        st.divider()
        st.markdown("### 📊 Reports")
        st.caption("Generate a comprehensive architecture document for the indexed repository.")

        if st.button("Generate Architecture Report", use_container_width=True):
            with st.spinner("Analyzing codebase and generating report (may take ~30 secs) ..."):
                try:
                    report = st.session_state.qa.generate_architecture_report()
                    st.session_state.arch_report = report
                    logger.info("Architecture report generated and stored in session state.")
                except Exception as e:
                    logger.exception("Architecture report generation failed: %s", e)
                    st.error(f"Report generation failed: {e}")

        if st.session_state.arch_report:
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=st.session_state.arch_report,
                file_name="architecture_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


# 4. Handle the repository ingestion process (cloning, chunking, and embedding)
if ingest_btn and repo_url:
    logger.info("Ingestion triggered for URL: %s (force=%s)", repo_url, force)
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        logger.error("Azure credentials missing from environment.")
        st.error("Azure credentials not found in environment. Please write them in your `.env` file.")
        st.stop()

    st.session_state.repo_url    = repo_url
    st.session_state.messages    = []
    st.session_state.summarized  = False
    st.session_state.qa          = None
    st.session_state.arch_report = ""

    with st.spinner("Cloning and indexing codebase (takes ~1-2 mins for medium repos) ..."):
        try:
            from ingest import ingest
            vs, repo_id, repo_dir = ingest(repo_url, force_reingest=force)
            st.session_state.repo_dir = repo_dir
        except Exception as e:
            logger.exception("Ingestion failed for %s: %s", repo_url, e)
            st.error(f"Ingestion failed: {e}")
            st.stop()

    with st.spinner("Building pinned repository card (~15 secs) ..."):
        try:
            from qa_engine import RepoQA
            qa = RepoQA(vs, repo_url=repo_url)
            qa.repo_dir = repo_dir          # expose repo dir to the engine
            st.session_state.qa      = qa
            st.session_state.repo_id = repo_id
        except Exception as e:
            logger.exception("Failed to initialise RepoQA for %s: %s", repo_url, e)
            st.error(f"Failed to generate repository card: {e}")
            st.stop()

    logger.info("Repository indexed successfully: %s (repo_id=%s)", repo_url, repo_id)
    st.success("Repository indexed successfully! Check out the dashboard below.")
    st.rerun()


# ── Helper: render source code expanders for files cited in an answer ──
def _render_source_viewer(sources: list[str], repo_dir: str) -> None:
    """Displays a collapsible expander for each cited source file."""
    if not sources:
        return

    # Auto-heal: derive repo_dir if missing in session state
    if not repo_dir and st.session_state.get("repo_id"):
        repo_dir = str(Path("./repos") / st.session_state.repo_id)
        st.session_state.repo_dir = repo_dir

    # Auto-heal: clone repo to ./repos/<repo_id> if directory doesn't exist on disk yet
    if repo_dir and not Path(repo_dir).exists() and st.session_state.get("repo_url"):
        try:
            from ingest import clone_repo, REPOS_DIR
            logger.info("Auto-cloning repo to %s for source viewer...", repo_dir)
            Path(REPOS_DIR).mkdir(parents=True, exist_ok=True)
            clone_repo(st.session_state.repo_url, repo_dir)
        except Exception as exc:
            logger.warning("Could not auto-clone for source viewer: %s", exc)

    if not repo_dir or not Path(repo_dir).exists():
        return

    valid_sources = [
        (src, Path(repo_dir) / src)
        for src in sources
        if (Path(repo_dir) / src).exists()
    ]
    if not valid_sources:
        return
    st.markdown("**📂 Referenced source files:**")
    for src, file_path in valid_sources:
        ext = file_path.suffix.lstrip(".")
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Could not read %s for source viewer: %s", src, exc)
            code = f"# Could not read file: {exc}"
        with st.expander(f"📄 `{src}`", expanded=False):
            st.code(code, language=ext or "text", line_numbers=True)


# --- INTERACTIVE Q&A CHAT ---
if not st.session_state.qa:
    st.info("Enter a public GitHub repository URL in the sidebar and click Load & Index Repository to start exploring.")

    st.markdown("""
    ### Try these popular public repositories:
    - `https://github.com/pallets/click` (Python CLI creation kit)
    - `https://github.com/tiangolo/fastapi` (FastAPI framework)
    - `https://github.com/requests/requests` (HTTP library)
    """)

else:
    # Show architecture report in main area if already generated
    if st.session_state.arch_report:
        with st.expander("📊 Architecture Report (click to expand / collapse)", expanded=False):
            st.markdown(st.session_state.arch_report)
        st.divider()

    col_chat, col_examples = st.columns([3, 1.2])

    with col_examples:
        st.markdown("### Quick Questions")
        st.caption("Click any question to ask it instantly:")

        examples = [
            ("Tech Stack & Summary", "What is the tech stack and main purpose of this repository?"),
            ("Key Entry Points", "Which files are the main entry points where execution starts?"),
            ("Project Structure", "Summarize the project structure and folder layout."),
            ("Q&A Engine Design", "How does the RepoQA class retrieve code and structure the context window?"),
            ("Noise Filtering", "What mechanisms are used to make sure noisy information is not added to the context?")
        ]

        clicked_question = None
        for label, question in examples:
            if st.button(label, key=f"btn_{label}", use_container_width=True):
                clicked_question = question

    with col_chat:
        st.markdown(f"**Current Repo:** `{st.session_state.repo_url}`")

        # Auto-onboard user with repository summary on first load
        if not st.session_state.summarized:
            st.session_state.summarized = True
            with st.chat_message("assistant"):
                st.write_stream(st.session_state.qa.summarize_repo())
            st.session_state.messages.append({
                "role": "assistant",
                "content": "(Project summary generated above)"
            })

        # Display past messages (and their source viewers if stored)
        for msg in st.session_state.messages:
            if msg["content"] == "(Project summary generated above)":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            # Re-render source viewer for assistant messages that have cited sources
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_source_viewer(msg["sources"], st.session_state.repo_dir)

        # Detect if user clicked an example question
        active_prompt = None
        if clicked_question:
            active_prompt = clicked_question

        # Or get text from standard chat input box
        if chat_prompt := st.chat_input("Ask anything about the repo ..."):
            active_prompt = chat_prompt

        if active_prompt:
            logger.info("User query received: %r", active_prompt[:120])
            st.session_state.messages.append({"role": "user", "content": active_prompt})
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                full_response = st.write_stream(st.session_state.qa.stream_answer(active_prompt))

            # Capture cited sources immediately after streaming completes
            cited_sources = list(st.session_state.qa.last_sources)

            # Render source viewer right after the response (before rerun)
            _render_source_viewer(cited_sources, st.session_state.repo_dir)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": cited_sources,   # persist so they re-render after rerun
            })
            st.rerun()