"""
query.py
========
Full RAG pipeline — 100% inside LlamaIndex:

  Settings.embed_model  → HuggingFace bge-small (local)
  Settings.llm          → OpenAILike pointing at HuggingFace router
                          (same pattern as official HF docs, wrapped in LlamaIndex)

  Pipeline:
    1. Reconnect to Supabase (no re-embedding)
    2. as_query_engine() handles: retrieve → prompt → LLM → answer
       all automatically through LlamaIndex

Install:
  pip install llama-index llama-index-vector-stores-supabase
               llama-index-embeddings-huggingface
               llama-index-llms-openai-like
               vecs python-dotenv
"""

import os
import sys
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ── 1. Load credentials ───────────────────────────────────────────────────────
load_dotenv()

DB_URL   = os.getenv("SUPABASE_DB_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not DB_URL:
    print("ERROR: SUPABASE_DB_URL not found in .env")
    sys.exit(1)
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env")
    print("Get a free token at: https://huggingface.co/settings/tokens")
    sys.exit(1)


# ── 2. Configure embedding + LLM via LlamaIndex Settings ──────────────────────
#
#   OpenAILike lets LlamaIndex talk to any OpenAI-compatible endpoint.
#   Here we point it at HuggingFace's router — exactly matching the
#   official HF docs pattern, but routed through LlamaIndex instead.
#
#   Official HF pattern:
#     OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
#     model="meta-llama/Llama-3.1-8B-Instruct:novita"
#
print("Configuring models...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"   # local, no API key, must match ingest.py
)

Settings.llm = OpenAILike(
    model="meta-llama/Llama-3.1-8B-Instruct:novita",
    api_base="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    max_tokens=512,
    temperature=0.2,
    is_chat_model=True,   # enables chat message formatting for Llama-3
)


# ── 3. Reconnect to Supabase — no re-embedding needed ─────────────────────────
print("Connecting to Supabase...")

vector_store = SupabaseVectorStore(
    postgres_connection_string=DB_URL,
    collection_name="dku_bulletin",   # must match ingest.py
    dimension=384,
)
index = VectorStoreIndex.from_vector_store(vector_store)
print("✅  Index reloaded from Supabase. No re-embedding performed.\n")


# ── 4. Custom RAG prompt ──────────────────────────────────────────────────────
#
#   {context_str} → filled automatically with the retrieved chunks
#   {query_str}   → filled automatically with the user's question
#
RAG_PROMPT = PromptTemplate(
    "You are a helpful academic advisor with expertise in the DKU Bulletin.\n"
    "Answer using ONLY the context below. "
    "If the answer is not in the context, say so clearly.\n\n"
    "Context:\n"
    "──────────────────────────────────────────────────────\n"
    "{context_str}\n"
    "──────────────────────────────────────────────────────\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


# ── 5. Build the query engine ─────────────────────────────────────────────────
#
#   as_query_engine() runs the full RAG loop automatically:
#     retrieve (Supabase) → build prompt → LLM (HF Llama-3.1) → answer
#
query_engine = index.as_query_engine(
    similarity_top_k=5,
    text_qa_template=RAG_PROMPT,
)


# ── 6. Ask a question ─────────────────────────────────────────────────────────
def ask(question: str) -> str:
    """
    Single call that runs the full RAG loop:
      embed question → retrieve from Supabase → inject context → Llama-3.1 answer

    Args:
        question: Natural language question about the DKU Bulletin.

    Returns:
        LLM-generated answer grounded in retrieved context.
    """
    print(f"Q: {question}")
    print("─" * 60)
    response = query_engine.query(question)
    answer = str(response).strip()
    print(f"A: {answer}")
    print("=" * 60 + "\n")
    return answer


# ── 7. Example questions ──────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What are the graduation requirements at DKU?",
        "What majors and concentrations are offered?",
        "What is the academic integrity policy?",
        "How does the grading system work?",
    ]

    for q in questions:
        ask(q)