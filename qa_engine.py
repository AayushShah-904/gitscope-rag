# This file runs our advanced Q&A engine. It uses "context engineering" to structure the AI's inputs,
# rewrites the user query with HyDE for better retrieval, fetches relevant chunks from Chroma using MMR,
# and formats the output cleanly for the user.

from __future__ import annotations

import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# Basic configurations for searching and history-keeping

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "github_repo"
TOP_K           = 8    # chunks returned to LLM
FETCH_K         = 24   # MMR candidate pool (3x TOP_K)
MAX_HISTORY     = 6    # conversation turn pairs kept in context


# Rules that teach the AI how to behave and formatting expectations

SYSTEM_PROMPT = """\
You are an expert software engineer assistant. Your job is to help users deeply \
understand a GitHub codebase.

Rules:
- Cite every file you reference using backtick paths, e.g. `src/auth/jwt.py`.
- When showing code, quote only the relevant lines from the provided context.
- If the context is insufficient to answer, say so clearly — never hallucinate code.
- For architecture questions, explain how components relate based on actual imports \
and call patterns you can see in the code.
- Use markdown. Keep answers focused — no padding.
"""


# Automatically detects the user's intent to suggest the best answer format

def _format_hint(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["explain", "how does", "how do", "what does", "why"]):
        return (
            "Structure your answer as: "
            "(1) plain-English explanation, "
            "(2) the key code snippet with file path, "
            "(3) which other files are involved."
        )
    if any(w in q for w in ["where", "which file", "find", "locate", "what file"]):
        return (
            "Lead with the exact file path(s), then a one-sentence explanation of each. "
            "Be concise."
        )
    if any(w in q for w in ["summarize", "overview", "architecture", "structure", "design"]):
        return (
            "Answer with a bullet-point summary grouped by layer or responsibility. "
            "Include file paths for each point."
        )
    if any(w in q for w in ["list", "what are", "show all", "enumerate"]):
        return "Answer as a numbered or bulleted list. Include file paths."
    if any(w in q for w in ["diff", "change", "break", "impact", "affect", "depend"]):
        return (
            "Answer by tracing the dependency chain: start from the changed item "
            "and list what calls or imports it, with file paths."
        )
    return "Answer concisely with file path citations. Use markdown."


# Helper functions to format documents and package context for the LLM

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        path = doc.metadata.get("source", "unknown")
        lang = doc.metadata.get("language", "")
        parts.append(f"### `{path}`\n```{lang}\n{doc.page_content}\n```")
    return "\n\n".join(parts)


def _log_context(messages: list[dict], label: str = "LLM CONTEXT") -> None:
    """
    Show the full messages payload to the logger at DEBUG level.
    Each message is printed with a role banner so the context window
    is easy to read in the log output.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    sep = "=" * 72
    lines = [f"\n{sep}", f"  {label}", sep]
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"\n--- Message {i} | role={role} ---")
        lines.append(content)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    lines.append(f"\n{sep}")
    lines.append(f"  Total messages: {len(messages)} | Total chars: {total_chars:,}")
    lines.append(sep)
    logger.debug("\n".join(lines))


def _build_messages(
    question: str,  
    context: str,
    repo_card: str,
    history: list[dict],
    format_hint: str,
) -> list[dict]:
    """
    Assembles the 6-layer context window recipe in order:
    1. System prompt (role and rules)
    2. Repository card (high-level overview pinned at top)
    3. Conversation history (sliding window of Q&As)
    4. Retrieved code chunks (relevance-filtered source code)
    5. Output format hint (intent-based formatting expectations)
    6. User question (latest request)
    """
    # LAYER 1: System instructions
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # LAYER 2: Pinned repo card (high-level overview)
    if repo_card:
        messages.append({
            "role": "user",
            "content": f"## Repository overview\n\n{repo_card}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the repository overview. Ask me anything.",
        })

    # LAYER 3: Slide-windowed conversation history
    messages.extend(history)

    # LAYER 4, 5, and 6: Code Context, Format Hints, and the current Question
    messages.append({
        "role": "user",
        "content": (
            f"## Relevant code from the repository\n\n{context}\n\n" # LAYER 4: Retrieved code
            f"---\n\n"
            f"**Output format:** {format_hint}\n\n"                   # LAYER 5: Output format hint
            f"**Question:** {question}"                              # LAYER 6: User question
        ),
    })

    _log_context(messages, label="_build_messages → LLM CONTEXT")
    return messages


# Prompts and tools to create a quick 150-word overview of the repo on first load

REPO_CARD_PROMPT = """\
Using the repository structure and README excerpts below, write a repo card \
in under 150 words covering exactly these four points:

1. What this project does (1-2 sentences)
2. Tech stack (comma-separated list)
3. Key entry points (file paths where execution starts)
4. Top-level folder purposes (one line each)

Be factual and concise. Use only what you can see in the provided context.

---

{context}
"""

def build_repo_card(vs: Chroma, llm: AzureChatOpenAI) -> str:
    logger.info("Building repository card ...")
    tree_docs   = vs.similarity_search("repository structure directory tree", k=1)
    readme_docs = vs.similarity_search("README project overview purpose install", k=3)
    context = _format_docs(tree_docs + readme_docs)

    repo_card_messages = [
        {"role": "system", "content": "You summarize software repositories concisely and accurately."},
        {"role": "user",   "content": REPO_CARD_PROMPT.format(context=context)},
    ]
    _log_context(repo_card_messages, label="build_repo_card → LLM CONTEXT")
    response = llm.invoke(repo_card_messages)
    logger.info("Repository card generated (%d chars).", len(response.content))
    return response.content


# HyDE (Hypothetical Document Embeddings) writes a fake answer first to find better search results

HYDE_PROMPT = """\
A developer is asking this question about a codebase:

"{question}"

Write a short hypothetical code snippet or technical explanation (3-6 sentences) \
that would directly answer this question. Be specific — include realistic function \
names, variable names, and file references. This will be used as a search query, \
not shown to the user.
"""

def _hyde_retrieve(question: str, vs: Chroma, llm: AzureChatOpenAI) -> list[Document]:
    try:
        logger.debug("HyDE: generating hypothetical answer for query: %r", question[:80])
        hypothetical = llm.invoke([
            {"role": "user", "content": HYDE_PROMPT.format(question=question)},
        ])
        search_text = hypothetical.content
        logger.debug("HyDE: hypothetical snippet generated (%d chars).", len(search_text))
    except Exception as exc:
        logger.warning("HyDE generation failed (%s); falling back to raw query.", exc)
        search_text = question  # graceful fallback

    raw_docs = vs.max_marginal_relevance_search(search_text, k=TOP_K, fetch_k=FETCH_K)
    
    # Filter out noisy or redundant documents (e.g. __directory_tree__, empty/whitespace chunks)
    filtered_docs = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "")
        # The directory tree is already summarized in the repo card; we exclude it here to avoid noise
        if source == "__directory_tree__" or not doc.page_content.strip():
            continue
        filtered_docs.append(doc)
    logger.debug(
        "Retrieved %d docs after MMR+filter (from pool of %d).",
        len(filtered_docs), len(raw_docs),
    )
    return filtered_docs


# Main Q&A engine class

class RepoQA:
    """
    Our main engine that handles searching the repository, remembering chat context,
    and generating clean answers for the user.
    """

    def __init__(self, vectorstore: Chroma, repo_url: str = "") -> None:
        self.vs       = vectorstore
        self.repo_url = repo_url

        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            streaming=True,
        )
        logger.info("RepoQA initialised for repo: %s", repo_url or "(unknown)")

        # [2] Built once, reused every call
        self.repo_card: str = build_repo_card(vectorstore, self.llm)

        # Holds a simple history of the conversation to handle follow-up questions
        self.history: list[dict] = []

    def _trim_history(self) -> list[dict]:
        return self.history[-(MAX_HISTORY * 2):]

    def _record_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

    def stream_answer(self, question: str) -> Iterator[str]:
        """Finds relevant code, streams the answer in real-time, and adds sources."""
        logger.info("stream_answer called | question: %r", question[:100])
        docs        = _hyde_retrieve(question, self.vs, self.llm)
        context     = _format_docs(docs)
        format_hint = _format_hint(question)
        messages    = _build_messages(
            question, context, self.repo_card, self._trim_history(), format_hint
        )

        full_response = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        sources = sorted({
            d.metadata.get("source", "")
            for d in docs
            if d.metadata.get("source") not in ("", "__directory_tree__")
        })
        if sources:
            citation_block = "\n\n---\n**Sources:**\n" + "\n".join(f"- `{s}`" for s in sources)
            yield citation_block
            full_response += citation_block

        logger.info(
            "stream_answer complete | %d source(s) cited | response length: %d chars",
            len(sources), len(full_response),
        )
        self._record_turn(question, full_response)

    def answer(self, question: str) -> tuple[str, list[Document]]:
        """Finds relevant code and returns the complete answer and source documents at once."""
        logger.info("answer called | question: %r", question[:100])
        docs        = _hyde_retrieve(question, self.vs, self.llm)
        context     = _format_docs(docs)
        format_hint = _format_hint(question)
        messages    = _build_messages(
            question, context, self.repo_card, self._trim_history(), format_hint
        )
        llm_sync = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            streaming=False,
        )
        response = llm_sync.invoke(messages)
        self._record_turn(question, response.content)
        return response.content, docs

    def summarize_repo(self) -> Iterator[str]:
        """Streams a helpful detailed summary of the repository for onboarding."""
        expand_prompt = (
            f"Based on this repo card:\n\n{self.repo_card}\n\n"
            f"Write a fuller onboarding summary for a new developer in this format:\n\n"
            f"**What this project does** (2-3 sentences)\n\n"
            f"**Tech stack** (bullet list)\n\n"
            f"**Key entry points** (file paths + one-line description each)\n\n"
            f"**Main modules / folders** (brief description of each top-level area)\n\n"
            f"**How to run it** (if you can infer from the repo card)\n\n"
            f"Keep it concise. Use only information from the repo card."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": expand_prompt},
        ]
        _log_context(messages, label="summarize_repo → LLM CONTEXT")
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history = []
        logger.info("Conversation history cleared for repo: %s", self.repo_url or "(unknown)")

    def generate_architecture_report(self) -> str:
        """
        Generates a comprehensive Architecture Document for the repository.
        Retrieves broad context from the vector store, then prompts the LLM
        to produce a detailed, well-structured markdown report.
        Returns the full report as a markdown string.
        """
        logger.info("Generating architecture report for repo: %s", self.repo_url or "(unknown)")

        # Pull broad context about structure, entry points, and key modules
        queries = [
            "repository architecture overview entry points main modules",
            "system design components dependencies data flow",
            "configuration environment setup deployment instructions",
            "API endpoints routes public interface",
        ]
        seen_sources: set[str] = set()
        all_docs: list[Document] = []
        for q in queries:
            for doc in _hyde_retrieve(q, self.vs, self.llm):
                src = doc.metadata.get("source", "")
                if src not in seen_sources:
                    seen_sources.add(src)
                    all_docs.append(doc)

        context = _format_docs(all_docs)

        report_prompt = f"""\
You are a senior software architect. Using the repository overview and source code below, \
write a comprehensive Architecture Document in markdown format.

## Repository Overview
{self.repo_card}

## Source Code Context
{context}

---

Generate the report with these exact sections:

# Architecture Document — {{repo_name}}

## 1. Executive Summary
A 3–5 sentence description of what the project does, its purpose, and its target users.

## 2. Tech Stack
A table with columns: Layer | Technology | Purpose

## 3. System Architecture
A high-level description of the system with ASCII art or a textual diagram showing how components interact.

## 4. Key Components
For each major file or module: file path, responsibility, and what it interacts with.

## 5. Data Flow
Step-by-step walkthrough of the primary use case (e.g. user makes a request → what happens next).

## 6. Directory Structure
Annotated version of the top-level folders with a one-line description of each.

## 7. Setup & Running
How to install dependencies and run the project (inferred from config files, README, etc.)

Be factual. Only use information from the provided context. Use proper markdown formatting.
Replace {{repo_name}} with the actual repository name from the URL or context.
"""

        messages = [
            {"role": "system", "content": "You are an expert software architect who writes clear, detailed architecture documents."},
            {"role": "user",   "content": report_prompt},
        ]
        _log_context(messages, label="generate_architecture_report → LLM CONTEXT")

        llm_sync = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0,
            streaming=False,
        )
        response = llm_sync.invoke(messages)
        logger.info("Architecture report generated (%d chars).", len(response.content))
        return response.content




# Helper function to easily initialize this class from our Streamlit app

def load_qa(repo_id: str, repo_url: str = "") -> RepoQA:
    collection = f"{COLLECTION_NAME}_{repo_id}"
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return RepoQA(vs, repo_url=repo_url)
