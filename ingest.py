"""
ingest.py
=========
Run ONCE: loads the DKU Bulletin PDF, embeds every chunk with a local
HuggingFace model, and persists all vectors to Supabase (pgvector).

After this script finishes, you never need to touch the PDF again.
Use query.py to retrieve answers at any time.
"""

import os
import sys
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ── 1. Load credentials from .env ────────────────────────────────────────────
load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
if not DB_URL:
    print("ERROR: SUPABASE_DB_URL not found.")
    print("Create a .env file with:  SUPABASE_DB_URL=postgresql://postgres:...")
    sys.exit(1)


# ── 2. Configure local embedding model (no API key required) ─────────────────
#
#   Model:  BAAI/bge-small-en-v1.5
#   Dims:   384  (must match the vector(384) column in Supabase)
#   Size:   ~130 MB — downloads automatically and is cached after first run
#
print("Loading embedding model (downloads ~130 MB on first run, cached after)...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = None  # No LLM needed for the indexing step


# ── 3. Parse the PDF into text chunks ────────────────────────────────────────
#
#   SimpleDirectoryReader uses pypdf under the hood.
#   Each page (or logical section) becomes one "Document" node.
#
PDF_PATH = "data/DKU_bulletin.pdf"
if not os.path.exists(PDF_PATH):
    print(f"ERROR: PDF not found at {PDF_PATH}")
    print("Place your DKU_bulletin.pdf inside the data/ folder.")
    sys.exit(1)

print(f"\nLoading PDF: {PDF_PATH}")
documents = SimpleDirectoryReader(
    input_files=[PDF_PATH]
).load_data()

print(f"  → {len(documents)} chunks extracted from PDF")


# ── 4. Connect to the Supabase vector store ───────────────────────────────────
#
#   SupabaseVectorStore wraps the 'vecs' library, which speaks directly
#   to the pgvector extension inside your Postgres database.
#
#   collection_name: maps to a table / collection in Supabase
#   dimension:       must match your embedding model (bge-small = 384)
#
print("\nConnecting to Supabase...")
vector_store = SupabaseVectorStore(
    postgres_connection_string=DB_URL,
    collection_name="dku_bulletin",   # will be created if it does not exist
    dimension=384,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# ── 5. Build the index: embed every chunk and write to Supabase ───────────────
#
#   This is the only slow step. For a 300-page PDF expect 2–5 minutes.
#   Progress is shown per-chunk.
#
print("\nEmbedding chunks and storing in Supabase...")
print("(This may take a few minutes for a large PDF — please wait)\n")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

print("\n" + "=" * 60)
print("✅  Ingestion complete!")
print("    All embeddings are now stored in Supabase.")
print("    Run query.py to retrieve answers — no PDF needed.")
print("=" * 60)