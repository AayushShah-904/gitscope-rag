# This file handles importing a GitHub repository.
# It clones the repo, cuts the code files into small pieces (chunking),
# converts them into numbers (embeddings), and saves them into our Chroma database.

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Generator

import git
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Basic settings for our database and folder paths

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "github_repo"
REPOS_DIR = "./repos"  # permanent home for cloned repositories

# Maps file extensions to their programming languages so we can split them correctly
LANGUAGE_MAP: dict[str, Language | None] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".ts":   Language.JS,       
    ".tsx":  Language.JS,
    ".jsx":  Language.JS,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".java": Language.JAVA,
    ".cpp":  Language.CPP,
    ".c":    Language.C,
    ".cs":   Language.CSHARP,
    ".rb":   Language.RUBY,
    ".md":   Language.MARKDOWN,
    ".mdx":  Language.MARKDOWN,
    ".html": Language.HTML,
    ".txt":  None,              # plain splitter
    ".env.example": None,
    ".yaml": None,
    ".yml":  None,
    ".json": None,
    ".toml": None,
}

# A list of folder names we want to skip (like virtual envs or cache folders)
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "vendor", "target",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "Cargo.lock",
    ".DS_Store", "thumbs.db",
}

MAX_FILE_BYTES = 150_000   # skip files larger than ~150 KB
CHUNK_SIZE     = 1_000     # tokens approx; LangChain uses chars internally
CHUNK_OVERLAP  = 150


# Useful helper functions for cloning, navigating, and building the repo structure

def clone_repo(url: str, target_dir: str) -> None:
    """Clones a GitHub repository using a shallow clone (depth=1) to save time and space."""
    logger.info("Cloning repository: %s", url)
    git.Repo.clone_from(url, target_dir, depth=1)
    logger.debug("Clone complete -> %s", target_dir)


def repo_id_from_url(url: str) -> str:
    """Generates a consistent short ID for the repository using its URL."""
    return hashlib.sha1(url.encode()).hexdigest()[:10]


def walk_repo(repo_dir: str) -> Generator[Path, None, None]:
    """Finds and lists all the code and text files in the repository that we care about."""
    root = Path(repo_dir)
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # Skip unwanted directories
        parts = set(path.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() not in LANGUAGE_MAP and path.name not in LANGUAGE_MAP:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            logger.debug("Skipping large file (>%d B): %s", MAX_FILE_BYTES, path.relative_to(root))
            continue
        yield path


def build_directory_tree(repo_dir: str, max_depth: int = 4) -> str:
    """
    Creates a text-based tree view of the folders and files.
    This gives the AI a bird's-eye view of how the repository is structured.
    """
    root = Path(repo_dir)
    lines = [root.name + "/"]

    def _recurse(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, entry in enumerate(entries):
            if entry.name in SKIP_DIRS or entry.name.startswith("."):
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _recurse(entry, prefix + extension, depth + 1)

    _recurse(root, "", 1)
    return "\n".join(lines)


def load_and_chunk_file(path: Path, repo_dir: str) -> list[Document]:
    """Reads a file and splits it into smaller parts using a language-specific splitter."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Could not read file %s: %s", path.name, e)
        return []

    if not content.strip():
        return []

    rel_path = str(path.relative_to(repo_dir))
    ext = path.suffix.lower()
    lang = LANGUAGE_MAP.get(ext) or LANGUAGE_MAP.get(path.name)

    # Choose the right splitting tool based on the programming language
    if lang is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    metadata = {
        "source": rel_path,
        "file_name": path.name,
        "language": lang.value if lang else "text",
        "extension": ext,
    }

    chunks = splitter.create_documents([content], metadatas=[metadata])

    # Save the chunk's index and total count so we can put them back together in order if needed
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


# Main entry point for ingestion

def ingest(repo_url: str, force_reingest: bool = False) -> tuple[Chroma, str, str]:
    """
    This is the main function that coordinates cloning, chunking, and saving
    the GitHub repository into our Chroma database.
    Returns (vectorstore, repo_id, repo_dir) where repo_dir is the permanent
    path to the cloned repository on disk (used by the source code viewer).
    """
    repo_id = repo_id_from_url(repo_url)
    collection = f"{COLLECTION_NAME}_{repo_id}"

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    # Permanent directory for this repo's cloned source files
    repo_dir = str(Path(REPOS_DIR) / repo_id)

    # Check if already ingested
    if not force_reingest and Path(CHROMA_DIR).exists():
        try:
            vs = Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
            chunk_count = vs._collection.count()
            if chunk_count > 0:
                logger.info(
                    "Collection '%s' already has %d chunks — skipping vector ingestion. "
                    "Set force_reingest=True to redo.",
                    collection, chunk_count,
                )
                if not Path(repo_dir).exists():
                    logger.info("Cloning source code to %s for source code viewer...", repo_dir)
                    Path(REPOS_DIR).mkdir(parents=True, exist_ok=True)
                    clone_repo(repo_url, repo_dir)
                return vs, repo_id, repo_dir
        except Exception:
            logger.debug("Could not read existing collection '%s'; will re-ingest.", collection)

    # If force re-ingesting, remove the old clone so we start fresh
    if force_reingest and Path(repo_dir).exists():
        logger.info("Force re-ingest: removing existing clone at %s", repo_dir)
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Clone into the permanent repo directory (skip cloning if already present)
    if not Path(repo_dir).exists():
        Path(REPOS_DIR).mkdir(parents=True, exist_ok=True)
        clone_repo(repo_url, repo_dir)
    else:
        logger.info("Repo already cloned at %s — skipping clone step.", repo_dir)

    # Phase 1: Build the directory structure map so the LLM knows where files live
    logger.info("Building directory tree ...")
    tree_text = build_directory_tree(repo_dir)
    tree_doc = Document(
        page_content=f"# Repository structure\n\n```\n{tree_text}\n```",
        metadata={"source": "__directory_tree__", "file_name": "__tree__", "language": "text", "extension": ""},
    )

    # Phase 2: Read, split, and organize the actual documentation and source code files
    logger.info("Walking and chunking repository files ...")
    all_docs: list[Document] = [tree_doc]
    file_count = 0

    for file_path in walk_repo(repo_dir):
        chunks = load_and_chunk_file(file_path, repo_dir)
        if chunks:
            all_docs.extend(chunks)
            file_count += 1

    logger.info("Processed %d files → %d total chunks", file_count, len(all_docs))

    # Convert chunks to vector numbers and save them in the database in batches of 500
    BATCH = 500
    logger.info("Embedding and storing chunks in Chroma (batch size=%d) ...", BATCH)
    vs = None
    for i in range(0, len(all_docs), BATCH):
        batch = all_docs[i : i + BATCH]
        if vs is None:
            vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection,
                persist_directory=CHROMA_DIR,
            )
        else:
            vs.add_documents(batch)
        logger.info("Stored %d / %d chunks ...", min(i + BATCH, len(all_docs)), len(all_docs))

    logger.info("Ingestion complete. %d chunks stored in collection '%s'.", len(all_docs), collection)
    return vs, repo_id, repo_dir


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/tiangolo/fastapi"
    vs, rid, rdir = ingest(url)
    logger.info("Vectorstore ready. Repo ID: %s | Repo Dir: %s", rid, rdir)