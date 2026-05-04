# FinDoc Agent

> Multimodal RAG agent for visually dense financial documents — annual reports, prospectuses, and research notes. Built on **ColQwen2** visual retrieval + **LangGraph** 7-node state machine with structured reflexion, grounding audit, and cross-turn memory.

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3c3c)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Chainlit](https://img.shields.io/badge/frontend-Chainlit-9c27b0)](https://docs.chainlit.io/)
[![ColQwen2](https://img.shields.io/badge/retriever-ColQwen2-ff6f00)](https://huggingface.co/vidore/colqwen2-v0.1)
[![Qdrant](https://img.shields.io/badge/vector_db-Qdrant-DC244C?logo=qdrant)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

**Language:** [简体中文](./README.zh-CN.md) | **English**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Key Design Decisions](#3-key-design-decisions)
4. [Project Structure](#4-project-structure)
5. [Quick Start](#5-quick-start)
6. [Tech Stack](#6-tech-stack)
7. [Roadmap](#7-roadmap)
8. [References](#8-references)

---

## 1. Overview

### The Problem

Financial annual reports present two fundamental challenges that break traditional text-chunking RAG pipelines:

1. **Visually complex, information-dense layouts.** Balance sheets, cash-flow statements, organizational charts, and footnote tables are spatially rich. OCR introduces errors; naive text chunking destroys table semantics and cross-cell relationships.
2. **Multi-hop, cross-document reasoning.** A typical user query — "compare the three-year gross margin trends of Company A and Company B" — requires multi-step retrieval, numeric computation, and comparative synthesis. Single-pass RAG cannot cover this.

### The Approach

**Use a vision-language retriever to bypass OCR entirely, and an agent state machine to orchestrate multi-step reasoning.**

- **ColQwen2** directly indexes *page images* with multi-vector embeddings + MaxSim late interaction, preserving table structure, chart layouts, and number formatting that OCR would corrupt.
- The **Agent** layer handles task decomposition, tool orchestration, root-cause diagnosis, differentiated remediation, and post-hoc citation/numeric auditing — transforming "retrieved pages" into "structured, cited, and verified answers."

### Capabilities (as of P25)

| Capability | Description |
|---|---|
| **Multi-document QA** | Query across 14+ indexed annual reports with automatic doc routing |
| **Comparative analysis** | Cross-company metric comparison with caliber disambiguation |
| **Numeric computation** | AST-safe expression evaluation (e.g., `(1505.6 + 1420.3) / 2`) |
| **Structured reflexion** | Root-cause-aware verification: `retrieval_miss` / `reading_miss` / `ambiguous_query` / `inconsistency` |
| **Grounding audit** | Post-hoc citation verification + numeric fuzzy match (±0.1%); unverified claims stripped with confidence banner |
| **Cross-turn memory** | Structured fact store (entity, period, metric, value) persists across conversation turns; follow-up questions skip retrieval |
| **Streaming output** | Token-by-token SSE streaming from synthesizer |
| **User document upload** | Upload PDFs/images → auto-encode → Qdrant index → immediately queryable |
| **Knowledge base management** | Panel UI for list / delete / reindex / cover preview |

---

## 2. Architecture

### 2.1 Agent Workflow (7-Node State Machine)

```
User Query
    │
    ▼
┌──────────────────┐  Pre-retrieval: MaxSim across all docs → top-3 candidates
│ retrieval_scout  │  (P22)
└────────┬─────────┘
         │
         ▼
┌──────────┐         Decomposes query → ordered SubTask list
│ Planner  │  ──►    Each SubTask: sub_query / target_doc / tool_calls / query_class
└──────────┘         Input includes candidate_docs + Available Tools registry
    │
    ▼
┌──────────┐         tool_calls → registry dispatch (P21)
│ Executor │  ──►    Fallback: expected_output_schema routing
└──────────┘         VLM output → fact_extractor structuring (P24)
    │                Pre-retrieval: check known_facts cache (P25)
    ▼
┌──────────┐         Structured missing_facts with root_cause + confidence
│ Verifier │  ──►    Early-stop: no new facts → force synthesizer
└──────────┘
    │
    ├─ sufficient ──────────────────┐
    │                                ▼
    ├─ not sufficient ─► ┌──────────┐  Per root_cause dispatch (P20):
    │                    │Remediation│  retrieval_miss → widen top_k, re-retrieve
    │                    └─────┬─────┘  reading_miss → refine instruction, re-read
    │                          │        ambiguous_query → rewrite self-contained
    │                          ▼        inconsistency → disambiguate_caliber
    │                      Executor     budget exhausted → fallthrough
    │                          │
    └──────────────────────────┘
                               ▼
                        ┌─────────────┐
                        │ Synthesizer │  Aggregate facts → cited answer (streaming)
                        └──────┬──────┘
                               ▼
                        ┌─────────────┐  Citation reverse-lookup + numeric fuzzy match
                        │  Grounding  │  Strip unverified claims + confidence banner
                        └─────────────┘
```

### 2.2 Deployment Topology

```
┌─ uvicorn backend.server:app (port 8001) ──────┐
│  FastAPI + SSE                                  │
│  POST /api/v1/query     → SSE stream (7 nodes) │
│  GET  /api/v1/documents → indexed doc list      │
│  GET  /api/v1/health                            │
│  startup → preload ColQwen2 + indexes           │
│  per-query → load/save conv_facts (P25)         │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        ▼                       ▼
┌────────────────┐    ┌──────────────────────┐
│ ColQwen Service│    │   Qdrant (MultiVector)│
│ Litserve + GPU │    │   Docker · port 6333  │
│ port 8000      │    │   collection:         │
└────────────────┘    │   findoc_pages         │
                      └──────────────────────┘

┌─ chainlit run app/chainlit_app.py ────────────┐
│  Pure UI (only imports chainlit / httpx)       │
│  Consumes SSE: event=node|status|token|done    │
│  Dynamic Step + inline Images + conf banner    │
└────────────────────────────────────────────────┘
```

**Key boundary:** `agent/`, `tools/`, and `ingestion/` are the business core, imported directly by the backend. The frontend communicates exclusively over HTTP/SSE and imports zero agent code.

### 2.3 Node Responsibilities

| Node | Role | Added |
|---|---|---|
| `retrieval_scout` | Pre-retrieves top-3 candidate docs with relevance scores; gives planner informed context | P22 |
| `planner` | Decomposes `question → [SubTask...]` with `tool_calls` / `target_doc` / `query_class` | P1 |
| `executor` | Dispatches SubTasks via Tool Registry or legacy schema routing; runs VLM reads concurrently | P1 + P21 |
| `verifier` | Structured sufficiency + consistency judgment → `MissingFact[]` with root cause | P1 + P19 |
| `remediation` | Dispatches 4 fix strategies per root cause with budget accounting | P20 |
| `synthesizer` | Composes final answer with `[doc_id p.N]` citations; streaming token output | P1 + P16 |
| `grounding` | Post-hoc audit: citation authenticity + numeric consistency; strips unverified claims | P23 |

### 2.4 Four Retriever Configurations

| `services.colqwen_url` | `retriever.backend` | Meaning |
|---|---|---|
| `""` | `in_memory` | Default dev mode: local model + local .pt MaxSim |
| `http://localhost:8000` | `in_memory` | Remote ColQwen + local MaxSim |
| `""` | `qdrant` | Local encoding + Qdrant server-side MaxSim |
| `http://localhost:8000` | `qdrant` | Fully distributed (recommended for production) |

Any Qdrant exception automatically fallbacks to `_in_memory_retrieve` — the agent never crashes on vector-DB failure.

---

## 3. Key Design Decisions

### 3.1 Why LangGraph over LangChain AgentExecutor?

LangChain's AgentExecutor is a black-box loop. LangGraph gives us an explicit state machine — every node's I/O is observable, reflexion loops are controlled via conditional edges, and the entire topology is 20 lines of assembly code in `graph.py`.

### 3.2 Why ColQwen2 for Visual Retrieval?

Traditional RAG: `PDF → OCR → text chunks → single-vector embedding → semantic search`. In financial documents, OCR error rates on dense tables are high, and single-vector compression loses spatial layout.

ColQwen2 encodes each page image into **multiple vectors** (one per patch, ~1024 vectors × 128 dims per page). Retrieval uses **MaxSim** — for each query token, find the best-matching document patch, then sum:

```
score(q, d) = Σ_{i ∈ q_tokens} max_{j ∈ d_tokens} ⟨q_i, d_j⟩
```

This preserves table structure, chart layouts, and number positions that OCR/text-chunking destroys. ViDoRe benchmark: nDCG@5 ~89% (ColPali ~81%). The 2B base model fits on an RTX 3060 6GB in bf16.

### 3.3 Why Qdrant?

Standard vector DBs (Chroma, Pinecone, pgvector) only support single-vector-per-document. A 200-page annual report produces ~200K vectors — Qdrant 1.10+ is one of the few databases with native `MultiVectorConfig(comparator=MAX_SIM)` support. We use `Distance.DOT` (not COSINE) to match Python's einsum dot-product behavior exactly.

### 3.4 Structured Reflexion (P19–P20)

Instead of a fuzzy "need more info" string, the verifier outputs structured `MissingFact` entries, each with a `root_cause` enum:

| root_cause | Fix Strategy |
|---|---|
| `retrieval_miss` | Widen `top_k`, re-retrieve with rewritten query |
| `reading_miss` | Refine VLM instruction, re-read same pages (no re-retrieval) |
| `ambiguous_query` | Rewrite to fully self-contained query |
| `inconsistency` | Trigger `disambiguate_caliber` to extract reporting caliber from each page |

Three layers of protection against infinite loops: `max_reflexion_iter=3` + `budget_retrievals=10` + `budget_vlm_calls=20`.

### 3.5 Tool Registry (P21)

Tools self-describe via `(name, description, params_schema, output_schema)`. The planner prompt auto-discovers available tools. The executor does pure dispatch + output validation. Adding a new tool requires one `register(ToolSpec(...))` call — no changes to planner, executor, or schemas.

| Tool | Category | Function |
|---|---|---|
| `retrieve_pages` | retrieval | ColQwen2 multi-vector MaxSim search |
| `read_page_with_vlm` | reading | VLM page image → structured text |
| `calculate` | compute | AST-bounded safe numeric evaluation |
| `disambiguate_caliber` | resolution | Cross-page number conflict → VLM caliber extraction |

### 3.6 Grounding Audit (P23)

Post-synthesis, pure-rule checks (regex + set lookups, zero LLM calls):
- **Citation check:** Every `[doc_id p.N]` in the answer must exist in `extracted_facts`
- **Numeric check:** Every number+unit pair must fuzzy-match a fact value (±0.1%)
- **Confidence banner:** ⚠ partial mismatch / 🛑 severe mismatch

### 3.7 Cross-Turn Fact Memory (P24–P25)

After each VLM read, `fact_extractor` (pure regex, zero LLM) extracts `(entity, period, metric, value, unit)` from Chinese financial text. Facts are persisted to SQLite per conversation. On follow-up questions, the executor checks `known_facts` first — if `(茅台, 2023, 营收)` is already known, skip retrieval+VLM entirely. Expected 40–60% reduction in retrieval calls for multi-turn sessions.

---

## 4. Project Structure

```
findoc-agent/
├── agent/                       # Agent core — 7-node LangGraph state machine
│   ├── graph.py                 # build_graph() / compile_graph()
│   ├── state.py                 # AgentState TypedDict + Fact / SubTask / PageHit
│   ├── schemas.py               # LLM structured output schemas (PlannerOutput, VerifierOutput)
│   ├── config.py                # config.yaml + env loader
│   ├── llm.py                   # ChatOpenAI factory (DeepSeek API)
│   ├── prompts/                 # Node prompt templates (.txt)
│   └── nodes/                   # 7 node implementations
│       ├── planner.py           #   retrieval_scout + planner (P22)
│       ├── executor.py          #   tool dispatch + VLM read + fact extraction (P21/P24/P25)
│       ├── verifier.py          #   sufficiency + consistency + structured missing_facts (P19)
│       ├── remediation.py       #   root-cause dispatch → 4 fix strategies (P20)
│       ├── synthesizer.py       #   cited answer + streaming token output (P16)
│       └── grounding.py         #   citation + numeric post-hoc audit (P23)
├── tools/                       # Tool layer — registry + 4 built-in tools
│   ├── registry.py              #   ToolSpec / REGISTRY / dispatch() (P21)
│   ├── colpali_tool.py          #   ColQwen2 retrieval (in-memory / Qdrant / remote)
│   ├── vlm_tool.py              #   VLM page reading (OpenAI-compat) + SQLite cache
│   ├── calculator.py            #   AST-bounded safe expression evaluator
│   ├── fact_extractor.py        #   Regex-based structured fact extraction (P24)
│   ├── disambiguate.py          #   Caliber disambiguation tool (P20)
│   └── vlm_cache.py             #   (image_path, instruction) → cached VLM output
├── ingestion/                   # Offline data pipeline
│   ├── pdf_to_pages.py          #   PDF → page PNGs
│   ├── build_index.py           #   ColQwen2 encode → .pt multi-vector index
│   ├── model_loader.py          #   Shared ColQwen2 model loading + encode logic
│   ├── push_to_qdrant.py        #   .pt → Qdrant upsert (idempotent)
│   └── upload.py                #   User upload pipeline (save → convert → encode → index)
├── services/                    # Model serving
│   └── colqwen_server.py        #   Litserve ColQwen2 GPU service
├── backend/                     # FastAPI backend
│   ├── server.py                #   POST /query SSE + CRUD + upload + conv_facts (P25)
│   ├── storage.py               #   SQLite (conversations / messages / documents / conv_facts)
│   └── schemas.py               #   API request/response models
├── app/                         # Frontend
│   ├── chainlit_app.py          #   Chainlit UI (SSE consumer + Step renderer)
│   └── data_layer.py            #   Chainlit DataLayer → backend SQLite
├── eval/                        # Evaluation
│   ├── qa_dataset.jsonl         #   QA pairs (30 planned, currently sample)
│   └── run_eval.py              #   Evaluation runner
├── config.yaml                  # Global config (model / retriever / services)
├── docker-compose.yml           # Qdrant container
└── requirements.txt
```

---

## 5. Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1 (for local ColQwen2 inference; optional if using remote service)
- Poppler (for `pdf2image`: `apt install poppler-utils` on Linux, `brew install poppler` on macOS)

### Setup

```bash
# 1. Environment
conda create -n findoc python=3.10 -y && conda activate findoc
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY and QWEN_API_KEY

# 3. Build index (first run)
python -m ingestion.pdf_to_pages --only "贵州茅台2023" --max_pages 5
python -m ingestion.build_index --only moutai_2023

# 4. Smoke test (runs with stub fallback when no keys)
python -m agent.graph
```

### Start Services

```bash
# (Optional) Start Qdrant for production multi-vector search
docker compose up -d qdrant
python -m ingestion.push_to_qdrant

# (Optional) Start ColQwen GPU service on a separate machine
python -m services.colqwen_server --port 8000

# Start backend
PYTHONPATH=. uvicorn backend.server:app --host 0.0.0.0 --port 8001 &

# Start frontend
chainlit run app/chainlit_app.py -w
```

### Configuration Matrix

Edit `config.yaml` to match your setup:

```yaml
retriever:
  backbone: colqwen2          # or colpali
  backend: in_memory          # or qdrant
  top_k: 5
services:
  colqwen_url: ""             # or http://gpu-host:8000
agent:
  max_reflexion_iter: 3
```

---

## 6. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Agent orchestration** | LangGraph | Explicit state machine, observable node I/O, conditional edges for reflexion |
| **LLM (text reasoning)** | DeepSeek V4 Flash | OpenAI-compatible API; Planner / Verifier / Synthesizer share one factory |
| **Visual retrieval** | ColQwen2 | Multi-vector + MaxSim; user LoRA backbone; fits RTX 3060 6GB in bf16 |
| **VLM (page reading)** | Qwen VLM (DashScope) | Page image → structured Chinese financial text |
| **Multi-vector DB** | Qdrant 1.13 | Native `MultiVectorConfig(comparator=MAX_SIM)`; `Distance.DOT` for einsum parity |
| **Model serving** | Litserve | vLLM does not support multi-vector encoders; Litserve is Python-native with GPU batching |
| **Backend** | FastAPI + SSE | Unidirectional streaming — no need for WebSocket overhead |
| **Frontend** | Chainlit | Python-native, LangGraph first-class citizen, zero changes to agent/tools/ingestion |

---

## 7. Roadmap

| Phase | Status | Description |
|---|---|---|
| P1–P4 | ✅ | Skeleton: directory layout, AgentState, LangGraph assembly, node/tool stubs, CLI smoke test |
| P5–P6 | ⏳ | Chainlit frontend + eval dataset (30 QA pairs) |
| P7–P10 | ✅ | ColQwen2 Litserve, Qdrant multi-vector, SSE progress streaming |
| P11–P18 | ✅ | VLM concurrency, VLM cache, conversation history, document upload, streaming output, auto-title, knowledge base panel |
| P19–P25 | ✅ | Structured verifier, differentiated remediation, Tool Registry, retrieval-scout planner, grounding audit, structured fact extraction, cross-turn memory |
| **P26** | ⏳ | **Error recovery:** full-chain retry + timeout + error_log |
| **P27** | ⏳ | **Context compression:** structured summarization over truncation |
| **P28** | ⏳ | **Memory upgrade:** semantic matching + 3-layer architecture (Working/Episodic/Semantic) |
| **P29** | ⏳ | **Task system:** DAG-based concurrent execution + plan_critic |
| **P30** | ⏳ | **Dynamic prompts:** query_class-driven prompt variants + few-shot injection |
| **P31** | ⏳ | **Multi-agent:** parallel verification + Supervisor/Specialist routing |
| **P32** | ⏳ | **Skill system:** reusable Tool+Prompt+Strategy capability units |

See [DEVLOG.md](./DEVLOG.md) for detailed engineering decisions and [LEARNLOG.MD](./LEARNLOG.MD) for core concepts deep-dive.

---

## 8. References

- **ColPali:** Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- **ColQwen2:** *Exploring Visual Language Models for Document Retrieval*, 2025 — [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1)
- **Reflexion:** Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- **MaxSim (Late Interaction):** Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Qdrant Multivector:** [qdrant.tech/documentation/concepts/vectors/#multivector](https://qdrant.tech/documentation/concepts/vectors/#multivector)

---

<p align="center">
  <sub>Language: <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a></sub>
</p>
