# FinDoc Agent

> Multimodal RAG agent for visually dense financial documents — annual reports, prospectuses, and research notes. Built on **ColQwen2** visual retrieval + **LangGraph** 8-node state machine with structured reflexion, LLMCompiler-style DAG execution, and cross-turn memory.

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
- The **Agent** layer handles query routing, task decomposition into DAG plans, LLMCompiler-style cross-task data flow, tool orchestration, root-cause diagnosis, differentiated remediation, and citation/numeric auditing — transforming "retrieved pages" into "structured, cited, and verified answers."

### Capabilities

| Capability | Description |
|---|---|
| **Query routing** | Keyword heuristics + lightweight LLM decides whether to engage retrieval or answer directly |
| **Multi-document QA** | Query across 14+ indexed annual reports with automatic doc routing |
| **Comparative analysis** | Cross-company metric comparison with caliber disambiguation |
| **Numeric computation** | AST-safe expression evaluation with `$task_id.value` cross-task data flow (LLMCompiler-style) |
| **Structured reflexion** | Root-cause-aware verification: `retrieval_miss` / `reading_miss` / `ambiguous_query` / `inconsistency` |
| **Citation verification** | Regex-based citation parsing from answer text; unverified `[doc p.N]` references stripped |
| **Cross-turn memory** | Three-layer architecture: Working dict → Episodic SQLite → Semantic global; cosine ≥ 0.85 hits skip retrieval |
| **Streaming output** | Token-by-token SSE streaming from synthesizer |
| **User document upload** | Upload PDFs/images → auto-encode → Qdrant index → immediately queryable |
| **Knowledge base management** | Panel UI for list / delete / reindex / cover preview |
| **Error recovery** | Tenacity exponential backoff, transient/fatal distinction, structured `error_log` |

---

## 2. Architecture

### 2.1 Agent Workflow (8-Node State Machine)

```
User Query
    │
    ▼
┌──────────────────┐  Keyword heuristic + LLM fallback (~80 tokens)
│  query_router    │  Decides: needs document retrieval?
└────────┬─────────┘
         │
   ┌─────┴─────┐
   │ needs     │
   │ retrieval?│
   └─────┬─────┘
   False │       True
         ▼         ▼
┌─────────────┐  ┌──────────────────┐  MaxSim across all docs → top-3 candidates
│ synthesizer │  │ retrieval_scout  │  (informs planner with candidate docs)
│  (direct)   │  └────────┬─────────┘
└──────┬──────┘           │
       │                  ▼
       │         ┌──────────┐         query_class localization → variant prompt
       │         │ planner  │  ──►    + few-shot → ordered DAG plan with task_ids
       │         └────┬─────┘         and `$tN.value` cross-task placeholders
       │              │
       │              ▼
       │         ┌──────────┐         DAG topological sort → ThreadPool concurrent
       │         │ executor │  ──►    layers; `$tN.value` resolution → calculator;
       │         └────┬─────┘         Tool Registry dispatch; VLM read + fact_extractor
       │              │
       │              ▼
       │    ┌────────────────────┐
       │    │ should trigger     │  Signal words or failed todos → plan_critic
       │    │ plan_critic?       │  revises plan (re-entry guard: cursor + iter cap)
       │    └──┬──────────┬──────┘
       │    No │          │ Yes
       │       ▼          ▼
       │  ┌──────────┐  ┌─────────────┐  Revises plan → loops back to executor
       │  │ verifier │  │ plan_critic │  (max 2 revisions per plan)
       │  └────┬─────┘  └──────┬──────┘
       │       │                │
       │       │                └──→ executor
       │       ├─ sufficient ──────────────────┐
       │       │                                ▼
       │       ├─ insufficient ─► ┌──────────┐  Root-cause dispatch → explicit tool_calls
       │       │                 │remediation│  (read_page_with_vlm / disambiguate_caliber)
       │       │                 └─────┬─────┘  Budget accounting; exhausted → fallthrough
       │       │                       │
       │       │                       ▼
       │       └─────────────────→ executor
       │                               │
       └───────────────────────────────┘
                                       ▼
                              ┌─────────────┐
                              │ synthesizer │  Aggregate facts → cited answer (SSE streaming)
                              └──────┬──────┘  Regex-parse `[doc p.N]` from output;
                                     │         strip fabricated citations against evidence
                                     ▼
                                     END
```

**8 nodes:** 7 permanent (query_router, retrieval_scout, planner, executor, verifier, remediation, synthesizer) + 1 on-demand (plan_critic).

### 2.2 Deployment Topology

```
┌─ uvicorn backend.server:app (port 8001) ──────┐
│  FastAPI + SSE                                  │
│  POST /api/v1/query     → SSE stream (8 nodes) │
│  GET  /api/v1/documents → indexed doc list      │
│  GET  /api/v1/health                            │
│  startup → preload ColQwen2 + indexes           │
│  per-query → load/save conv_facts               │
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
│  Consumes SSE: event=status|token|node|todo    │
│  Dynamic Step + inline Images + TodoList       │
└────────────────────────────────────────────────┘
```

**Key boundary:** `agent/`, `tools/`, and `ingestion/` are the business core, imported directly by the backend. The frontend communicates exclusively over HTTP/SSE and imports zero agent code.

### 2.3 Node Responsibilities

| Node | Role |
|---|---|
| `query_router` | Keyword heuristic + ~80 token LLM decide retrieval vs direct answer; conditional edge routes accordingly |
| `retrieval_scout` | Pre-retrieves top-3 candidate docs with relevance scores; gives planner informed context |
| `planner` | Two-stage: `query_class` localization → variant prompt + few-shot → ordered DAG plan with `task_id`, `depends_on`, `$tN.value` placeholders |
| `executor` | DAG topological scheduling + ThreadPool concurrency; resolves `$tN.value` placeholders → calculator; Tool Registry dispatch; VLM reads |
| `plan_critic` | On-demand plan revision triggered by signal words or failed todos; re-entry protection via `plan_critic_last_cursor` + `plan_critic_iter` cap (max 2) |
| `verifier` | Structured `MissingFact[]` with 4 root-cause types; numeric/comparative queries: 3-instance parallel voting (strict/base/numeric) |
| `remediation` | Root-cause dispatch → explicit tool_calls (not string-prefix hacks); 3-budget protection (iter=3 / retrieval=10 / vlm=20) |
| `synthesizer` | Composes final answer with `[doc p.N]` citations; streaming token output; regex-parses citations → strips fabricated refs against evidence set |

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

LangChain's AgentExecutor is a black-box loop. LangGraph gives us an explicit state machine — every node's I/O is observable, reflexion loops are controlled via conditional edges, and the entire topology is ~30 lines of assembly code in `graph.py`.

### 3.2 Why Not ReAct?

Financial QA is structured (entity × period × metric). ReAct's serial think-act-observe loop loses parallelism in cross-company comparisons, burns tokens, and kills observability. A decomposition-then-execute DAG with typed node outputs is a better fit.

### 3.3 Why ColQwen2 for Visual Retrieval?

Traditional RAG: `PDF → OCR → text chunks → single-vector embedding → semantic search`. In financial documents, OCR error rates on dense tables are high, and single-vector compression loses spatial layout.

ColQwen2 encodes each page image into **multiple vectors** (one per patch, ~1024 vectors × 128 dims per page). Retrieval uses **MaxSim** — for each query token, find the best-matching document patch, then sum:

```
score(q, d) = Σ_{i ∈ q_tokens} max_{j ∈ d_tokens} ⟨q_i, d_j⟩
```

This preserves table structure, chart layouts, and number positions that OCR/text-chunking destroys. ViDoRe benchmark: nDCG@5 ~89% (ColPali ~81%). The 2B base model fits on an RTX 3060 6GB in bf16.

### 3.4 Why Qdrant?

Standard vector DBs (Chroma, Pinecone, pgvector) only support single-vector-per-document. A 200-page annual report produces ~200K vectors — Qdrant 1.10+ is one of the few databases with native `MultiVectorConfig(comparator=MAX_SIM)` support. We use `Distance.DOT` (not COSINE) to match Python's einsum dot-product behavior exactly.

### 3.5 Query Routing

Not all queries need retrieval. "Hello" / "What can you do?" / follow-up questions answerable from chat history should skip the entire retrieval+plan+execute pipeline. `query_router` uses keyword heuristics for strong signals (e.g., "2023年营收" → retrieve; "你好" → direct), falling back to a ~80 token LLM call for ambiguous queries. This avoids wasting retrieval+VLM budget on casual turns.

### 3.6 LLMCompiler-Style DAG Execution

Planner outputs a DAG plan — each subtask has a `task_id` and `depends_on` list. Cross-task data flows through `$tN.value` placeholders: earlier tasks' computed values are substituted before downstream tasks invoke the calculator. The executor does topological scheduling with same-layer ThreadPool concurrency. Synthesizer prompt enforces a hard rule: "if a `compute:` line exists, use its value — no recalculation."

### 3.7 Structured Reflexion

Instead of a fuzzy "need more info" string, the verifier outputs structured `MissingFact` entries, each with a `root_cause` enum:

| root_cause | Fix Strategy |
|---|---|
| `retrieval_miss` | Widen `top_k`, re-retrieve with rewritten query |
| `reading_miss` | Construct explicit `read_page_with_vlm` tool_calls for the same pages |
| `ambiguous_query` | Rewrite to fully self-contained query |
| `inconsistency` | Trigger `disambiguate_caliber` with conflicting fact texts |

Three layers of protection against infinite loops: `max_reflexion_iter=3` + `budget_retrievals=10` + `budget_vlm_calls=20`.

### 3.8 Tool Registry

Tools self-describe via `(name, description, params_schema, output_schema)`. The planner prompt auto-discovers available tools. The executor does pure dispatch + output validation. Adding a new tool requires one `register(ToolSpec(...))` call — no changes to planner, executor, or schemas.

| Tool | Category | Function |
|---|---|---|
| `retrieve_pages` | retrieval | ColQwen2 multi-vector MaxSim search |
| `read_page_with_vlm` | reading | VLM page image → structured text |
| `calculate` | compute | AST-bounded safe numeric evaluation with `$tN.value` resolution |
| `disambiguate_caliber` | resolution | Cross-page number conflict → VLM caliber extraction |

### 3.9 Citation Verification

Post-synthesis, pure-rule checks (regex + set lookups, zero LLM calls): citations are parsed from answer text via `[doc p.N]` pattern matching, then checked against the `extracted_facts` evidence set. Fabricated references are stripped. Only pages actually cited are returned to the frontend.

### 3.10 Cross-Turn Fact Memory (Three-Layer)

- **Working memory**: `fact_index` dict `{(entity, period, metric): Fact}` — lives within a single query
- **Episodic memory**: SQLite `conv_facts` table + ColQwen text encoder 128d float16 embeddings — persists across conversation turns; `known_facts` checked before retrieval+VLM
- **Semantic memory**: facts with `hit_count ≥ 3` and `grounding_verified=1` are promoted to `global_facts` for cross-conversation reuse

On follow-up questions, the executor checks `known_facts` first — if `(茅台, 2023, 营收)` is already known, skip retrieval+VLM entirely. Expected 40–60% reduction in retrieval calls for multi-turn sessions.

---

## 4. Project Structure

```
findoc-agent/
├── agent/                       # Agent core — 8-node LangGraph state machine
│   ├── graph.py                 #   build_graph() — 30-line topology with conditional edges
│   ├── state.py                 #   AgentState TypedDict + Fact / SubTask / PageHit / Citation
│   ├── schemas.py               #   LLM structured output schemas (PlannerOutput, VerifierOutput)
│   ├── config.py                #   config.yaml + env loader
│   ├── llm.py                   #   ChatOpenAI factory (DeepSeek API)
│   ├── compression.py           #   TokenBudget-aware context summarization
│   ├── memory.py                #   Three-layer memory: Working / Episodic / Semantic
│   ├── retry.py                 #   Tenacity exponential backoff + transient/fatal classification
│   ├── prompts/                 #   Node prompt templates (.txt) + few-shot examples (.jsonl)
│   └── nodes/                   #   8 node implementations
│       ├── query_router.py      #     Keyword heuristic + LLM fallback → route retrieval vs direct
│       ├── planner.py           #     retrieval_scout + planner (two-stage with query_class)
│       ├── executor.py          #     DAG scheduling + $tN.value resolution + tool dispatch + fact extraction
│       ├── plan_critic.py       #     On-demand plan revision (signal-word / failed-todo trigger)
│       ├── verifier.py          #     Structured MissingFact + 3-instance parallel voting
│       ├── remediation.py       #     Root-cause dispatch → explicit tool_calls + budget accounting
│       └── synthesizer.py       #     Cited answer + streaming + citation regex parsing
├── tools/                       # Tool layer — registry + 4 built-in tools
│   ├── registry.py              #     ToolSpec / REGISTRY / dispatch()
│   ├── colpali_tool.py          #     ColQwen2 retrieval (in-memory / Qdrant / remote)
│   ├── vlm_tool.py              #     VLM page reading (OpenAI-compat) + SQLite cache
│   ├── calculator.py            #     AST-bounded safe expression evaluator
│   ├── fact_extractor.py        #     Regex-based structured fact extraction
│   ├── disambiguate.py          #     Caliber disambiguation tool
│   └── vlm_cache.py             #     (image_path, instruction) → cached VLM output
├── skills/                      # Skill system — reusable Tool+Prompt+Strategy units
│   ├── registry.py              #     YAML skill loader + trigger-keyword matching
│   ├── single_fact.yaml         #     Single-fact query skill profile
│   ├── multi_step_calc.yaml     #     Multi-step calculation skill profile
│   ├── cross_doc_compare.yaml   #     Cross-document comparison skill profile
│   └── trend_analysis.yaml      #     Trend analysis skill profile
├── ingestion/                   # Offline data pipeline
│   ├── pdf_to_pages.py          #     PDF → page PNGs
│   ├── build_index.py           #     ColQwen2 encode → .pt multi-vector index
│   ├── model_loader.py          #     Shared ColQwen2 model loading + encode logic
│   ├── push_to_qdrant.py        #     .pt → Qdrant upsert (idempotent)
│   └── upload.py                #     User upload pipeline (save → convert → encode → index)
├── services/                    # Model serving
│   └── colqwen_server.py        #     Litserve ColQwen2 GPU service
├── backend/                     # FastAPI backend
│   ├── server.py                #     POST /query SSE + CRUD + upload + conv_facts
│   ├── storage.py               #     SQLite (conversations / messages / documents / conv_facts)
│   └── schemas.py               #     API request/response models
├── app/                         # Frontend
│   ├── chainlit_app.py          #     Chainlit UI (SSE consumer + Step renderer + TodoList)
│   └── data_layer.py            #     Chainlit DataLayer → backend SQLite
├── eval/                        # Evaluation
│   ├── queries.yaml             #     QA pairs (3 questions, target 30)
│   └── run_eval.py              #     Evaluation runner
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
| **Error recovery** | Tenacity | Exponential backoff; transient retry vs fatal immediate-fail distinction |

---

## 7. Roadmap

| Phase | Status | Description |
|---|---|---|
| P1–P4 | ✅ | Skeleton: directory layout, AgentState, LangGraph assembly, node/tool stubs, CLI smoke test |
| P5–P6 | ⏳ | Chainlit frontend + eval dataset (30 QA pairs — currently 3 questions) |
| P7–P10 | ✅ | ColQwen2 Litserve, Qdrant multi-vector, SSE progress streaming |
| P11–P18 | ✅ | VLM concurrency, VLM cache, conversation history, document upload, streaming output, auto-title, knowledge base panel |
| P19–P25 | ✅ | Structured verifier, differentiated remediation, Tool Registry, retrieval-scout planner, grounding audit, structured fact extraction, cross-turn memory |
| **P26** | ✅ | **Error recovery:** full-chain retry + timeout + structured `error_log` |
| **P27** | ✅ | **Context compression:** structured summarization + TokenBudget management |
| **P28** | ✅ | **Memory upgrade:** three-layer architecture (Working/Episodic/Semantic) + cosine semantic matching |
| **P29** | ✅ | **DAG execution:** LLMCompiler-style `$tN.value` data flow + plan_critic on-demand revision |
| **P30** | ✅ | **Dynamic prompts:** query_class-driven prompt variants + few-shot injection (3 DAG examples) |
| **P31** | ✅ | **Multi-agent:** 3-instance parallel verification (strict/base/numeric majority vote) |
| **P32** | ✅ | **Skill system:** YAML-based Tool+Prompt+Strategy capability units with trigger-keyword matching |

See [DEVLOG.md](./DEVLOG.md) for detailed engineering decisions and change logs, and [LEARNLOG.MD](./LEARNLOG.MD) for core concepts deep-dive.

---

## 8. References

- **ColPali:** Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- **ColQwen2:** *Exploring Visual Language Models for Document Retrieval*, 2025 — [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1)
- **LLMCompiler:** Kim et al., *An LLM Compiler for Parallel Function Calling*, ICML 2024
- **Reflexion:** Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- **MaxSim (Late Interaction):** Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Qdrant Multivector:** [qdrant.tech/documentation/concepts/vectors/#multivector](https://qdrant.tech/documentation/concepts/vectors/#multivector)

---

<p align="center">
  <sub>Language: <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a></sub>
</p>
