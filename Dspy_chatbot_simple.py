"""
dspy_chatbot.py
===============
Simple DSPy RAG chatbot over the DKU Bulletin using HuggingFace Inference API.

How dspy.LM connects to HuggingFace (from official DSPy docs):
  "If your provider offers an OpenAI-compatible endpoint,
   just add an openai/ prefix to your full model name."

  lm = dspy.LM(
      model    = "openai/<model-name>",   # openai/ prefix = OpenAI-compatible
      api_base = "https://router.huggingface.co/v1",
      api_key  = HF_TOKEN,
  )

This is the same HuggingFace router URL used in query.py — just wired
through dspy.LM instead of the openai Python client directly.

Install:
  pip install dspy llama-index llama-index-vector-stores-supabase
               llama-index-embeddings-huggingface vecs python-dotenv
"""

import os
import sys
import time
from dotenv import load_dotenv

import dspy
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy.exc import OperationalError


# ── 1. Load credentials ───────────────────────────────────────────────────────
load_dotenv()

DB_URL   = os.getenv("SUPABASE_DB_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not DB_URL:
    print("ERROR: SUPABASE_DB_URL not found in .env"); sys.exit(1)
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env"); sys.exit(1)


# ── 2. Configure dspy.LM with HuggingFace Inference API ──────────────────────
#
#   DSPy uses LiteLLM under the hood for all LM calls.
#   LiteLLM supports any OpenAI-compatible endpoint via the "openai/" prefix.
#
#   This is exactly the same as query.py's:
#     OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
#     model="meta-llama/Llama-3.1-8B-Instruct:novita"
#
#   — just expressed as dspy.LM instead.
#
lm = dspy.LM(
    model    = "openai/meta-llama/Llama-3.1-8B-Instruct:novita",
    api_base = "https://router.huggingface.co/v1",
    api_key  = HF_TOKEN,
    max_tokens  = 512,
    temperature = 0.2,
)
dspy.configure(lm=lm)
print("✅  dspy.LM configured → Llama-3.1-8B via HuggingFace router")


# ── 3. Load embedding model + reconnect to Supabase ───────────────────────────
#
#   LlamaIndex handles embeddings + Supabase retrieval exactly as in query.py.
#   DSPy only takes over for the generation step.
#
print("Loading embedding model and connecting to Supabase...")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None  # LlamaIndex LLM disabled — dspy.LM handles generation

print("✅  Supabase index reloaded. No re-embedding performed.\n")


# ── 4. DSPy-compatible retriever wrapping LlamaIndex + Supabase ───────────────
#
#   ROOT CAUSE of the error:
#     Supabase (and most Postgres hosts) close idle connections after a timeout.
#     The vecs/SQLAlchemy connection pool holds a stale connection, and the next
#     query fails with "server closed the connection unexpectedly".
#
#   FIX:
#     Catch OperationalError, rebuild the entire VectorStore + retriever from
#     scratch (forces a fresh TCP connection), then retry the query once.
#     This is the safest approach because vecs doesn't expose pool.reset().
#
def _build_retriever(db_url: str, k: int):
    """Create a fresh LlamaIndex retriever with a new Supabase connection."""
    vs = SupabaseVectorStore(
        postgres_connection_string=db_url,
        collection_name="dku_bulletin",
        dimension=384,
    )
    idx = VectorStoreIndex.from_vector_store(vs)
    return idx.as_retriever(similarity_top_k=k)


class SupabaseRM:
    """
    Wraps the LlamaIndex + Supabase retriever for DSPy.
    Auto-reconnects if Supabase drops the idle connection.
    """
    def __init__(self, db_url: str, k: int = 5):
        self.db_url = db_url
        self.k = k
        self._retriever = _build_retriever(db_url, k)

    def _reconnect(self):
        print("\n[Supabase] Connection lost — reconnecting...", flush=True)
        time.sleep(1)
        self._retriever = _build_retriever(self.db_url, self.k)
        print("[Supabase] Reconnected.\n", flush=True)

    def __call__(self, query: str, k: int = None) -> list[str]:
        self._retriever.similarity_top_k = k or self.k
        try:
            nodes = self._retriever.retrieve(query)
        except (OperationalError, Exception) as e:
            # Catch stale-connection errors and retry once with a fresh connection
            if "server closed the connection" in str(e) or "OperationalError" in type(e).__name__:
                self._reconnect()
                nodes = self._retriever.retrieve(query)  # retry
            else:
                raise
        return [node.text for node in nodes]


retriever = SupabaseRM(db_url=DB_URL, k=5)


# ── 5. DSPy Signature ─────────────────────────────────────────────────────────
#
#   Signatures define what goes IN and what comes OUT of the LLM call.
#   DSPy automatically turns this into a well-structured prompt —
#   no manual prompt writing needed.
#
class AnswerFromBulletin(dspy.Signature):
    """
    Answer student questions about the DKU Bulletin accurately and concisely.
    Base your answer only on the provided context.
    If the answer is not in the context, say so clearly.
    """
    context:  list[str] = dspy.InputField(desc="Relevant passages from the DKU Bulletin")
    question: str        = dspy.InputField(desc="Student's question about DKU")
    answer:   str        = dspy.OutputField(desc="Accurate answer grounded in the context")


# ── 6. DSPy RAG Module ────────────────────────────────────────────────────────
#
#   dspy.Module wires retrieval + generation into one reusable component.
#   ChainOfThought adds step-by-step reasoning before the final answer.
#
class DKUBulletinRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(AnswerFromBulletin)

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1: retrieve from Supabase via LlamaIndex embeddings
        context = retriever(question)

        # Step 2: dspy.LM (HuggingFace) generates a grounded answer
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(context=context, answer=prediction.answer)


# ── 7. Instantiate ────────────────────────────────────────────────────────────
rag = DKUBulletinRAG()


# ── 8. Chat loop ──────────────────────────────────────────────────────────────
def chat():
    print("=" * 60)
    print("  DKU Bulletin Chatbot  (DSPy + HuggingFace + Supabase)")
    print("  Commands:  'quit' to exit | 'history' to inspect last LM call")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # 'history' shows the exact prompt DSPy sent to the LM — useful for debugging
        if question.lower() == "history":
            dspy.inspect_history(n=1)
            continue

        print("\nBot: ", end="", flush=True)
        result = rag(question=question)
        print(result.answer)
        print()


if __name__ == "__main__":
    chat()