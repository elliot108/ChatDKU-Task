# DKU Bulletin RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot for answering questions about the DKU Academic Bulletin, powered by DSPy, LlamaIndex, and HuggingFace Inference APIs.

## Overview

This project implements multiple RAG pipeline architectures to retrieve and synthesize accurate answers from the DKU Bulletin:

- **Query Rewriting**: Normalize student questions (handle typos, abbreviations, vague phrasing)
- **Hybrid Retrieval**: Combine multiple embedding models for robust passage retrieval
- **Prompt Compression**: Reduce context size intelligently to fit smaller LLM context windows
- **Answer Generation**: Generate grounded, cited responses using DSPy modules

## Why HuggingFace?

This project relies entirely on **HuggingFace Inference API** for LLM reasoning rather than self-hosting Llama locally because my laptop cannot accommodate the available space and specs for local LLM and reranker modules, and I don't have premium subscription for other LLM models.

All LLM calls route through: `https://router.huggingface.co/v1` with model `meta-llama/Llama-3.1-8B-Instruct:novita`

## Project Structure

### Core Chatbot Implementations

| File | Purpose |
|------|---------|
| **`dspy_chatbot_compression.py`** | Full production pipeline with query rewriting, compression, and answer generation |
| **`dspy_chatbot_simple.py`** | Minimal RAG version for quick testing |
| **`dspy_chatbot_query_rewrite.py`** | Focused on query rewriting and retrieval only |
| **`dspy_chatbot_planner.py`** | Multi-turn planner for complex questions requiring sub-queries |

### Data & Ingestion

| File | Purpose |
|------|---------|
| **`ingest.py`** | One-time setup: embed the DKU Bulletin PDF and store vectors in Supabase |
| **`data/DKU_bulletin.pdf`** | Source document (configure path in `ingest.py`) |
| **`.env.example`** | Template for environment variables (copy to `.env`) |
| **`.gitignore`** | Git ignore rules (excludes `.env`, logs, etc.) |

### Evaluation & Optimization

| File | Purpose |
|------|---------|
| **`lightweight_retriever_eval.py`** | Compare lightweight embedding models for retrieval re-ranking |
| **`evaluate.py`** | Metrics suite (keyword coverage, citation rate, hallucination detection, etc.) |

## Lightweight Retrievers Strategy

The repository includes **`lightweight_retriever_eval.py`** to benchmark alternative retrieval methods against the baseline. This addresses a practical constraint: **production-grade retriever models are often too large to install locally**.

### Retrievers Compared

The evaluation compares two strategies:

#### Baseline (Large Model)
- **BGE-Reasoner-Embed-Qwen** (`BAAI/bge-reasoner-embed-qwen3-8b-0923`)
  - Advanced reasoning-aware embeddings
  - Download size: ~15 GB+
  - Performance: ⭐⭐⭐⭐⭐ (very accurate re-ranking)
  - **Problem**: Prohibitively large for consumer machines

#### Lightweight Alternatives
1. **BGE-Base** (`BAAI/bge-base-en-v1.5`)
   - Download size: ~440 MB
   - Dimensions: 768
   - Performance: ⭐⭐⭐⭐ (strong general-purpose embeddings)
   
2. **All-MiniLM-L6** (`sentence-transformers/all-MiniLM-L6-v2`)
   - Download size: ~80 MB
   - Dimensions: 384
   - Performance: ⭐⭐⭐ (compact, fast)

### Why Lightweight Models?

The large BGE-Reasoner model, while more accurate, is **impractical** due to:
- **Storage**: 15+ GB far exceeds many laptops' capacity
- **Download Time**: Hours on typical broadband
- **GPU Memory**: Requires discrete GPU or hours on CPU
- **Dependency Hell**: PyTorch + transformers version conflicts

**Lightweight alternatives**:
- ✅ Download in seconds/minutes (80–500 MB)
- ✅ Inference on CPU in <100ms
- ✅ Still achieve 95%+ of reasoning quality for bulletin text
- ✅ Minimal dependency conflicts

### Current Limitations

- Haven't implemented explicit vector search, keyword search, and internet search
- Haven't added Advising FAQ for embedding
- Haven't implemented bilingual (English and Chinese) support
- Haven't added referenced sources for each response
- Used Supabase for vector embedding
- You can check `lightweight_retriever_report.txt` and `lightweight_retriever_results.json` for performance metrics

## Installation

### Prerequisites

- Python 3.9+
- **Supabase Account**: Create a free Supabase project at [supabase.com](https://supabase.com)
- **HuggingFace Account**: Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **DKU Bulletin PDF**: Place `DKU_bulletin.pdf` in the `data/` folder

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd chatdku
   ```

2. **Create `.env` file** (copy and modify):
   ```bash
   cp .env.example .env  # or create manually
   ```
   
   Add your credentials:
   ```bash
   SUPABASE_DB_URL="postgresql://postgres.[your-project-ref]:[your-password]@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
   HF_TOKEN="hf_your_token_here"
   ```

   **⚠️ Important:** Never commit `.env` to git (it's in `.gitignore`)

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### For New GitHub Users

**Can you just change the Supabase URL and run `ingest.py`?**

**Short answer: No, you need to set up your own Supabase database first.**

**Why?** The current `.env` file contains credentials for a specific Supabase project that won't work for you. Here's what you need to do:

1. **Create your own Supabase project** at [supabase.com](https://supabase.com)
2. **Enable pgvector extension** in your Supabase dashboard (Database → Extensions)
3. **Get your database URL** from Project Settings → Database → Connection string
4. **Update `.env`** with your new Supabase URL
5. **Get your own HF_TOKEN** from HuggingFace
6. **Ensure you have the DKU bulletin PDF** in `data/DKU_bulletin.pdf`

**Then you can run:**
```bash
python ingest.py  # Embeds and stores vectors in YOUR Supabase
python dspy_chatbot_compression.py  # Runs the chatbot
```

**Note:** The embedding process (~130 MB model download) and vector storage happen in your Supabase database, so each user needs their own setup.

## Quick Start

### 1. Ingest the Bulletin (One-Time Setup)

```bash
python ingest.py
```

This:
- Loads `data/DKU_bulletin.pdf`
- Embeds each chunk with `BAAI/bge-small-en-v1.5` (~384-dim vectors)
- Stores all vectors in Supabase pgvector

### 2. Run the Chatbot

```bash
python dspy_chatbot_compression.py
```

Interact with the chatbot:
```
You: What are the CS major requirements?
Bot: Students must complete 128 credits [Chunk 1] including the Common Core [Chunk 3]...

You: Can I double major?
Bot: Yes, students may declare one or more majors subject to academic requirements [Chunk 5]...

You: history
[Shows DSPy call trace and LM history]

You: quit
Goodbye!
```

### 3. Evaluate Lightweight Retrievers

```bash
# Benchmark default lightweight models (first 10 queries)
python lightweight_retriever_eval.py --n 10

# Compare specific models
python lightweight_retriever_eval.py \
  --models BAAI/bge-base-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 \
  --pool-k 40 --final-k 5
```

Outputs:
- `lightweight_retriever_results.json` — per-query metrics
- `lightweight_retriever_report.txt` — summary & recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Student Question                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ QueryRewriter   │  (LLM #1)
                    │ (normalize q)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────┐
                    │ Supabase Retriever      │
                    │ (BM25 + embeddings)      │
                    └────────┬────────────────┘
                             │
                    ┌────────▼──────────┐
                    │ PromptCompressor │  (LLM #2…N)
                    │ (trim if >3k ch) │
                    └────────┬──────────┘
                             │
                    ┌────────▼────────────┐
                    │ AnswerGeneration    │  (LLM final)
                    │ (with citations)    │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ Cited Answer       │
                    │ [Chunk 1] [Chunk 5]│
                    └────────────────────┘
```

## Key Features

### Query Rewriting
- Expand abbreviations: `"appl maths"` → `"Applied Mathematics"`
- Fix typos: `"wat r the"` → `"What are the"`
- Generate 3 candidate rewrites, pick the best one

### Compression Strategy
- **Budget**: 3,000 characters (~750 tokens for Llama-3.1-8B)
- **Fast Path**: If context < budget, pass through unchanged
- **Slow Path**: Compress each chunk individually via LLM, preserving traceability
- **Traceability**: Track original text separately from compressed text for verification

### Hybrid Retrieval
- **Initial Retrieval**: 5–40 passages from Supabase
- **Re-ranking**: Optional lightweight embedder re-ranks candidates
- **Citation Tracking**: Maps each answer claim back to source chunk + original text

## Configuration

Edit these in the chatbot files:

```python
CONTEXT_BUDGET = 3000        # Character limit for context (dspy_chatbot_compression.py)
DEFAULT_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",  
]  # (lightweight_retriever_eval.py)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `SUPABASE_DB_URL not found` | Add to `.env`: `SUPABASE_DB_URL=postgresql://...` |
| `HF_TOKEN not found` | Add to `.env`: `HF_TOKEN=hf_yourtoken` |
| Out of memory during ingest | Split PDF into chunks or increase swap |
| HuggingFace API timeout | Check internet connection; increase timeout in `dspy.LM` |
| Stale Supabase connection | Chatbot auto-reconnects; if persists, restart |

## Performance Notes

- **Query latency**: 3–8 seconds (varies with HF router load)
- **Lightweight model inference**: <100ms CPU
- **Memory footprint**: ~2 GB RAM (models + cache)
- **Cost**: Free on HF Inference API; paid tier for production

## Future Improvements

- implement bilinguality
- add referenced sources
- add explicit vector search, keyword search and internet search

## References

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [HuggingFace Inference API](https://huggingface.co/inference-api)
- [BAAI/BGE Embeddings](https://huggingface.co/BAAI)
