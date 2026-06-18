# FAQ Semantic Search Engine with ChromaDB and LLM-Powered Answer Generation

## Overview

This project implements an intelligent FAQ retrieval system using semantic search and Large Language Models (LLMs). Instead of relying on traditional keyword matching, the system converts FAQ entries into vector embeddings and stores them in ChromaDB for efficient similarity search.

When a user submits a query, the engine retrieves the most relevant FAQs and uses them as context for an LLM to generate accurate, grounded responses with source attribution and confidence scores.

This project demonstrates the core concepts behind Retrieval-Augmented Generation (RAG) systems.

---

## Features

* Semantic search using vector embeddings
* ChromaDB vector database integration
* Metadata-based FAQ categorization
* Top-k FAQ retrieval
* LLM-powered grounded answer generation
* Source attribution with similarity scores
* Out-of-scope query detection
* Modular and extensible architecture

---

## Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* GPT-OSS / OpenAI-compatible LLM
* NumPy

---

## Project Architecture

User Query
↓
Embedding Model
↓
ChromaDB Vector Search
↓
Top-K Relevant FAQs
↓
LLM Context Construction
↓
Grounded Answer Generation
↓
Answer + Source Attribution

---

## Dataset Structure

Each FAQ entry contains:

```python
{
    "question": "...",
    "answer": "...",
    "category": "..."
}
```

Categories include:

* Account
* Billing
* Security
* Technical

---

## Example Queries

### Password Reset

Query:

```text
I forgot my login credentials
```

Retrieved FAQ:

```text
How do I reset my password?
```

### Subscription Cancellation

Query:

```text
I want to stop my subscription
```

Retrieved FAQ:

```text
How do I cancel my subscription?
```

### API Rate Limits

Query:

```text
How many API calls can I make per hour?
```

Retrieved FAQ:

```text
What is the API rate limit?
```

---

## Future Improvements

* Hybrid Search (Keyword + Semantic Search)
* Category-Based Filtering
* Conversation Memory
* Multi-Language Support
* Web Interface using Streamlit
* Full RAG Pipeline Integration
* Real Customer Support Dataset

---

## Learning Outcomes

Through this project, I learned:

* Vector embeddings
* Semantic similarity search
* ChromaDB indexing and retrieval
* Metadata filtering
* Context-aware answer generation
* Fundamentals of Retrieval-Augmented Generation (RAG)

---


