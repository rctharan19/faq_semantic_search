# 🔍 FAQ Semantic Search Engine

A production-ready FAQ retrieval system powered by **semantic search** and **LLM-grounded answer generation** — the foundational pattern behind Retrieval-Augmented Generation (RAG).

---

## 📌 What This Does

Traditional keyword search fails when users phrase questions differently from how FAQs are written. This system solves that by understanding *meaning*, not just matching words.

| Step | What Happens |
|------|-------------|
| **Index** | FAQ entries are embedded using sentence-transformers and stored in ChromaDB |
| **Retrieve** | User query is embedded and compared via cosine similarity to find top-3 FAQs |
| **Generate** | Retrieved FAQs are passed as context to an LLM to produce a grounded answer |
| **Attribute** | Each response includes source FAQs with confidence scores |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Embedding Model]         ← sentence-transformers (local)
    │
    ▼
[ChromaDB Vector Store]   ← cosine similarity search → Top-K FAQs
    │
    ▼
[LLM via OpenRouter]      ← context-grounded generation
    │
    ▼
Answer + Source Attribution
```

This is the **RAG (Retrieval-Augmented Generation)** pattern — retrieve relevant context first, then generate, rather than relying on the LLM's parametric memory alone.

---

## 🗂️ Project Structure

```
faq-semantic-search/
├── faq_search.py        # Main pipeline (index → retrieve → generate)
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/faq_semantic_search.git
cd faq_semantic_search
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
```bash
cp .env.example .env
# Edit .env and paste your OpenRouter API key
```

Or export it directly:
```bash
export OPENROUTER_API_KEY=your_key_here   # macOS/Linux
set OPENROUTER_API_KEY=your_key_here      # Windows CMD
```

> Get a free API key at [openrouter.ai](https://openrouter.ai)

---

## ▶️ Run

```bash
python faq_search.py
```

**Sample output:**
```
🚀  FAQ Semantic Search Engine — RAG Foundation Demo

✅  Indexed 10 FAQs into ChromaDB collection 'faq_knowledge_base'

============================================================
🔍  Query: I forgot my password, what should I do?
============================================================

📚  Top-3 Retrieved FAQs:
  1. [account   ]  confidence=0.9312  ██████████████████
     Q: How do I reset my password?
     A: Go to the login page and click 'Forgot Password'...

🤖  Generated Answer:
  To reset your password, go to the login page and click
  'Forgot Password'. Enter your registered email and you'll
  receive a reset link valid for 24 hours. [Source: FAQ 1]

📌  Source Attribution:
  [1] "How do I reset my password?"  (confidence: 0.9312)
```

---

## 🧰 Tech Stack

| Component | Library / Service |
|-----------|------------------|
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | `sentence-transformers` (via ChromaDB default EF) |
| LLM | [OpenRouter](https://openrouter.ai) — any OSS model |
| Similarity | Cosine distance (HNSW index) |
| Language | Python 3.9+ |

---

## 🔧 Customise

- **Swap the LLM**: Change `MODEL_NAME` in `faq_search.py` to any model on OpenRouter (e.g. `google/gemma-2-9b-it`, `meta-llama/llama-3-8b-instruct`).
- **Add your own FAQs**: Edit the `FAQ_DATABASE` list — add as many `{question, answer, category}` dicts as needed.
- **Filter by category**: Pass `where={"category": "billing"}` to `collection.query()` for category-scoped search.
- **Persist the index**: Replace `chromadb.Client()` with `chromadb.PersistentClient(path="./chroma_db")` to avoid re-indexing on every run.

---

## 📚 Concepts Demonstrated

- **Semantic embeddings** — representing meaning as vectors
- **Vector similarity search** — cosine distance in high-dimensional space
- **Metadata filtering** — ChromaDB `where` clauses for scoped retrieval
- **Context-grounded generation** — preventing hallucination by anchoring the LLM to retrieved facts
- **Source attribution** — confidence scores for retrieved evidence
- **RAG pipeline foundation** — retrieve → augment → generate

---
---

