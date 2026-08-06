<div align="center">

# 🔭 GitScope RAG

### AI-powered codebase intelligence — understand any GitHub repository in minutes

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-F97316?style=flat-square)](https://trychroma.com)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4.1-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[**Live Demo →**](https://share.streamlit.io) &nbsp;|&nbsp; [**Report a Bug**](https://github.com/AayushShah-904/gitscope-rag/issues) &nbsp;|&nbsp; [**Request a Feature**](https://github.com/AayushShah-904/gitscope-rag/issues)

</div>

---

## 📖 Overview

**GitScope RAG** is an AI-powered code analysis tool that lets you deeply understand any public GitHub repository through natural language conversation. Paste a GitHub URL, wait ~2 minutes for indexing, and start asking questions like:

- *"How does authentication work in this codebase?"*
- *"What is the tech stack and main purpose of this project?"*
- *"Which files are the main entry points?"*
- *"How do these components interact with each other?"*

Built using **Retrieval-Augmented Generation (RAG)** with advanced context engineering techniques including **HyDE** (Hypothetical Document Embeddings) and **MMR** (Maximal Marginal Relevance) for precise, grounded answers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Smart Retrieval** | HyDE + MMR search for high-precision, diverse code chunk retrieval |
| 🧠 **6-Layer Context Window** | Structured context engineering: system prompt → repo card → history → code → format hint → question |
| 📁 **Multi-Language Support** | Handles Python, JavaScript, TypeScript, Go, Rust, Java, C++, C, C#, Ruby, Markdown, and more |
| 💬 **Conversational Memory** | Sliding window of up to 6 conversation turns for accurate follow-up questions |
| 🗂️ **Repo Card** | Auto-generated pinned summary (tech stack, entry points, folder layout) on every session |
| ⚡ **Streaming Answers** | Real-time streamed responses with file path citations |
| 📐 **Intent-Based Formatting** | Automatically detects question type and adapts output format |
| 🏗️ **Architecture Reports** | Generate comprehensive architecture documents for any repo |
| 🔄 **Smart Caching** | Skips re-ingestion if a repo is already indexed; force re-ingest with a checkbox |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│      Sidebar: Repo URL Input + Ingest Button             │
│      Main: Chat Interface + Quick Question Buttons       │
└──────────────────┬───────────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   ingest.py        │
         │  1. Shallow clone  │
         │  2. Walk & filter  │
         │  3. Language-aware │
         │     chunking       │
         │  4. Embed & store  │
         │     → ChromaDB     │
         └─────────┬─────────┘
                   │  vectorstore
         ┌─────────▼──────────────────┐
         │       qa_engine.py         │
         │  RepoQA class:             │
         │  1. Build Repo Card (LLM)  │
         │  2. HyDE query expansion   │
         │  3. MMR retrieval          │
         │  4. 6-layer context build  │
         │  5. Stream answer (LLM)    │
         └────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Azure OpenAI     │
         │  GPT-4.1-mini      │
         │  text-embedding    │
         │  -ada-002          │
         └────────────────────┘
```

### Context Window Layers

The `_build_messages()` function assembles a precise 6-layer prompt for every query:

```
Layer 1 │ System Prompt       → Role + behavioural rules
Layer 2 │ Repo Card           → Pinned high-level overview (built once, reused)
Layer 3 │ Conversation History → Sliding window of last 6 Q&A turns
Layer 4 │ Retrieved Code      → HyDE+MMR fetched chunks (TOP_K=8)
Layer 5 │ Format Hint         → Intent-based output structure guidance
Layer 6 │ User Question       → The actual query
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35+ | Interactive web UI & streaming chat |
| **Orchestration** | LangChain 0.2+ | Document loading, text splitting, chain management |
| **Vector Store** | ChromaDB 0.5+ | Local persistent vector database |
| **Embeddings** | Azure OpenAI `text-embedding-ada-002` | Code chunk vectorization |
| **LLM** | Azure OpenAI `gpt-4.1-mini` | Answer generation, HyDE, repo card |
| **Repo Cloning** | GitPython 3.1+ | Shallow git clone of public repositories |
| **Code Splitting** | LangChain `RecursiveCharacterTextSplitter` | Language-aware chunking |
| **Config** | python-dotenv | Environment variable management |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- An **Azure OpenAI** account with:
  - A chat model deployment (e.g. `gpt-4.1-mini`)
  - An embeddings model deployment (e.g. `text-embedding-ada-002`)
- Git installed on your system

### 1. Clone the Repository

```bash
git clone https://github.com/AayushShah-904/gitscope-rag.git
cd gitscope-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-ada-002
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Enter a public GitHub URL** in the sidebar (e.g. `https://github.com/pallets/click`)
2. Click **"Load & Index Repository"** — the app will clone, chunk, and embed the codebase (~1–2 min)
3. Once indexed, a **repo card summary** is automatically generated
4. **Chat naturally** or click any of the **Quick Question** buttons on the right panel
5. Use **"Clear Conversation"** to start a fresh session on the same repo
6. Check **"Force re-ingest"** to re-clone and re-embed an already-indexed repo

### Example Questions

```
What is the tech stack and main purpose of this repository?
Which files are the main entry points where execution starts?
How does authentication work in this codebase?
Summarize the project structure and folder layout.
What mechanisms are used to filter noisy information from context?
```

---

## 📂 Project Structure

```
gitscope-rag/
├── app.py              # Streamlit UI — page config, sidebar, chat interface
├── ingest.py           # Repo cloning, file walking, chunking, and ChromaDB ingestion
├── qa_engine.py        # RepoQA class — HyDE retrieval, context building, streaming
├── requirements.txt    # Python dependencies
├── packages.txt        # System-level packages for Streamlit Cloud (git)
├── .gitignore          # Excludes .env, chroma_db/, venv/, repos/
└── .env                # ⚠️ Local only — never committed
```

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud** for free.

### Deploy Your Own Instance

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your fork → set main file to `app.py`
4. In **Advanced settings → Secrets**, add your Azure credentials in TOML format:

```toml
AZURE_OPENAI_API_KEY = "..."
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1-mini"
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-ada-002"
```

5. Click **"Deploy!"**

> **Note:** Streamlit Community Cloud uses ephemeral storage. The `chroma_db` index is cleared on server restarts. Simply re-enter the repository URL to re-index.

---

## ⚙️ Configuration

Key tunable parameters in `ingest.py` and `qa_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Approximate characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `MAX_FILE_BYTES` | `150,000` | Max file size to process (~150 KB) |
| `TOP_K` | `8` | Number of chunks returned to the LLM |
| `FETCH_K` | `24` | MMR candidate pool size (3× TOP_K) |
| `MAX_HISTORY` | `6` | Conversation turn pairs kept in context |

---

## 🗺️ Roadmap

- [ ] Support for private repositories (via GitHub PAT)
- [ ] Persistent vector store (e.g. Pinecone, Qdrant) for cross-session memory
- [ ] Downloadable PDF architecture reports
- [ ] Multi-repo comparison mode
- [ ] OpenAI / Anthropic / Gemini provider support alongside Azure

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ by <a href="https://github.com/AayushShah-904">Aayush Shah</a>
</div>
