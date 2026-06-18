"""
FAQ Semantic Search Engine
==========================
A retrieval-augmented system that indexes FAQ entries in ChromaDB,
performs semantic search on user queries, and generates grounded
answers using an LLM — the foundational pattern for RAG pipelines.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI  # OpenRouter-compatible client

# ── Configuration ───────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-7b-instruct"   # swap freely on OpenRouter
TOP_K = 3                                       # FAQs retrieved per query
COLLECTION_NAME = "faq_knowledge_base"

# ── FAQ Knowledge Base ───────────────────────────────────────────────────────

FAQ_DATABASE = [
    {
        "question": "How do I reset my password?",
        "answer":   "Go to the login page and click 'Forgot Password'. Enter your registered email. "
                    "You will receive a reset link valid for 24 hours.",
        "category": "account"
    },
    {
        "question": "Can I change my email address?",
        "answer":   "Yes. Go to Settings → Account → Change Email. Verification is required on both "
                    "the old and new email addresses.",
        "category": "account"
    },
    {
        "question": "How do I cancel my subscription?",
        "answer":   "Navigate to Settings → Billing → Cancel Plan. Your access continues until the "
                    "end of the current billing period. No partial refunds.",
        "category": "billing"
    },
    {
        "question": "What payment methods do you accept?",
        "answer":   "We accept all major credit/debit cards (Visa, Mastercard, Amex), UPI, "
                    "net banking, and PayPal.",
        "category": "billing"
    },
    {
        "question": "How do I get an invoice for my payment?",
        "answer":   "Invoices are auto-generated after each payment. Download them from "
                    "Settings → Billing → Invoice History.",
        "category": "billing"
    },
    {
        "question": "Is my data backed up?",
        "answer":   "Yes. All data is backed up daily to geographically distributed servers. "
                    "Backups are retained for 30 days.",
        "category": "security"
    },
    {
        "question": "Do you comply with GDPR?",
        "answer":   "Yes. We are fully GDPR-compliant. You can request a data export or "
                    "deletion at any time from Settings → Privacy.",
        "category": "security"
    },
    {
        "question": "What is the API rate limit?",
        "answer":   "Free tier: 100 requests/hour. Pro: 1,000/hour. Enterprise: unlimited. "
                    "Rate limit headers are included in every API response.",
        "category": "technical"
    },
    {
        "question": "How do I get an API key?",
        "answer":   "Go to Settings → Developer → API Keys. Click 'Generate New Key'. "
                    "Keep it secret — it has full account access.",
        "category": "technical"
    },
    {
        "question": "How do I add a team member?",
        "answer":   "Go to Settings → Team → Invite Member. Enter their email and choose a role "
                    "(Admin, Editor, Viewer). They will receive an invitation email.",
        "category": "account"
    },
]

# ── Step 1: Build ChromaDB Index ─────────────────────────────────────────────

def build_index() -> chromadb.Collection:
    """
    Embeds every FAQ entry and stores it in a ChromaDB collection
    with category metadata for optional filtered retrieval.
    """
    client = chromadb.Client()

    # Use the default sentence-transformer embedding model
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Recreate collection (idempotent for demos)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    documents, metadatas, ids = [], [], []
    for idx, faq in enumerate(FAQ_DATABASE):
        # Index the question — that's what users' queries resemble
        documents.append(faq["question"])
        metadatas.append({
            "answer":   faq["answer"],
            "category": faq["category"],
            "question": faq["question"],
        })
        ids.append(f"faq_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅  Indexed {len(FAQ_DATABASE)} FAQs into ChromaDB collection '{COLLECTION_NAME}'")
    return collection


# ── Step 2: Semantic Retrieval ───────────────────────────────────────────────

def retrieve_faqs(collection: chromadb.Collection, query: str, top_k: int = TOP_K):
    """
    Embeds the user query and finds the top_k most similar FAQ questions.
    Returns a list of dicts with question, answer, category, and distance score.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    hits = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        confidence = round(1 - dist, 4)   # cosine distance → similarity score
        hits.append({
            "question":   meta["question"],
            "answer":     meta["answer"],
            "category":   meta["category"],
            "confidence": confidence,
        })
    return hits


# ── Step 3: Grounded Answer Generation ──────────────────────────────────────

def generate_answer(query: str, retrieved_faqs: list) -> str:
    """
    Sends the top retrieved FAQs as context to the LLM and asks it to
    generate a concise, grounded answer — core RAG pattern.
    """
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    context_block = "\n\n".join(
        f"FAQ {i+1} (category: {faq['category']}, confidence: {faq['confidence']}):\n"
        f"Q: {faq['question']}\nA: {faq['answer']}"
        for i, faq in enumerate(retrieved_faqs)
    )

    system_prompt = (
        "You are a helpful support assistant. Answer the user's question using ONLY "
        "the FAQ context provided. If the context doesn't cover the question, say so. "
        "Be concise and cite which FAQ(s) you drew from."
    )

    user_prompt = (
        f"User question: {query}\n\n"
        f"Relevant FAQs:\n{context_block}\n\n"
        "Generate a clear, grounded answer."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# ── Step 4: Full Pipeline ────────────────────────────────────────────────────

def faq_search(query: str, collection: chromadb.Collection):
    """
    End-to-end FAQ semantic search:
      1. Retrieve top-K semantically similar FAQs
      2. Feed them as context to the LLM
      3. Print the grounded answer + source attribution
    """
    print(f"\n{'='*60}")
    print(f"🔍  Query: {query}")
    print(f"{'='*60}")

    # Retrieval
    hits = retrieve_faqs(collection, query)

    print(f"\n📚  Top-{TOP_K} Retrieved FAQs:")
    for i, hit in enumerate(hits, 1):
        bar = "█" * int(hit["confidence"] * 20)
        print(f"  {i}. [{hit['category']:10}]  confidence={hit['confidence']:.4f}  {bar}")
        print(f"     Q: {hit['question']}")
        print(f"     A: {hit['answer'][:80]}{'...' if len(hit['answer']) > 80 else ''}")

    # Generation
    print(f"\n🤖  Generated Answer:")
    answer = generate_answer(query, hits)
    print(f"  {answer}")

    print(f"\n📌  Source Attribution:")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] \"{hit['question']}\"  (confidence: {hit['confidence']:.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🚀  FAQ Semantic Search Engine — RAG Foundation Demo")
    print(f"    Database: {len(FAQ_DATABASE)} entries | Top-K: {TOP_K} | Model: {MODEL_NAME}\n")

    # Build the vector index once
    collection = build_index()

    # Sample queries to demonstrate the system
    queries = [
        "I forgot my password, what should I do?",
        "How can I download my billing receipt?",
        "Is the platform secure? What about GDPR?",
        "How many API calls can I make per hour?",
    ]

    for query in queries:
        faq_search(query, collection)

    # Interactive mode
    print(f"\n{'='*60}")
    print("💬  Interactive Mode  (type 'quit' to exit)")
    print(f"{'='*60}")
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input:
            faq_search(user_input, collection)


if __name__ == "__main__":
    main()
