# Multi-Stage RAG System for NeurIPS Competition

This repository contains our submission for the NeurIPS RAG competition. The system addresses the challenge of combining real-time information with archival knowledge through a three-stage retrieval and synthesis pipeline.

## Problem Statement

Modern question-answering systems face a fundamental tension: LLMs have outdated knowledge cutoffs, while pure web search lacks depth and context. We needed an approach that could leverage both live web data and comprehensive archival sources, then synthesize them coherently.

## Our Approach

### Stage 1: Query Decomposition and Live Retrieval

We start by breaking down complex queries into focused sub-questions using Phi-3-mini. This decomposition is critical because monolithic queries often miss nuanced aspects of multi-faceted questions.

**Why Phi-3-mini?** We chose this model for its efficiency and strong instruction-following capabilities. The 4k context window is sufficient for query expansion while maintaining fast inference times.

The live scraping component uses Playwright to pull from Google News and Bing. We deliberately chose headless browser automation over APIs to avoid rate limiting and maintain flexibility across different search engines.

```python
# Example: "What is happening in Nepal?" becomes:
# 1. What are the recent political developments in Nepal?
# 2. What is the current economic situation in Nepal?
# 3. Are there any natural disasters or events in Nepal recently?
```

### Stage 2: Archival Retrieval with FAISS

For each sub-question, we query the FineWeb dataset and build a FAISS index on-the-fly. This might seem inefficient compared to maintaining a persistent index, but it offers significant advantages for the competition setting:

- **Query-specific indexing**: We only index documents relevant to the current query, reducing noise
- **Fresh retrieval**: Each query gets tailored document selection from FineWeb
- **Memory efficiency**: No need to maintain massive persistent indices

We use `all-mpnet-base-v2` for embeddings because it strikes the right balance between semantic understanding and computational cost. The chunking strategy (200 words, 50-word overlap) ensures we don't lose context at boundaries while keeping chunks semantically coherent.

**Design decision**: We chunk before embedding rather than embedding full documents because it allows for finer-grained retrieval. A document might contain one relevant paragraph among many irrelevant ones.

### Stage 3: Context Synthesis

The final stage merges live web context with archival chunks. We explicitly demarcate the two sources in the prompt:

```
[START OF LIVE WEB CONTEXT]
...
[END OF LIVE WEB CONTEXT]

[START OF ARCHIVAL CONTEXT]
...
[END OF ARCHIVAL CONTEXT]
```

This structure helps the LLM understand temporal context and source reliability. Live web data is more current but potentially less reliable, while archival data provides depth and verification.

We use Phi-3-mini again for synthesis to maintain consistency and because it handles structured prompts well. The instruction to "not invent facts" combined with clearly marked source boundaries reduces hallucination risk.

## Key Technical Choices

**Async operations**: Stage 1 uses `asyncio` and `nest_asyncio` because web scraping is I/O-bound. Parallel requests significantly reduce latency.

**JSON output parsing**: Query expansion outputs JSON with regex fallback. LLMs sometimes add markdown formatting or explanatory text, so we extract the first valid JSON object rather than failing on malformed output.

**No persistent storage**: Everything is computed per-query. For a competition submission, this ensures reproducibility and avoids the complexity of database management.

## Installation

```bash
pip install nest-asyncio playwright transformers torch sentence-transformers faiss-cpu requests
playwright install chromium
```

## Usage

```python
from retrieval.stage1 import stage1
from retrieval.stage2 import stage2
from retrieval.stage3 import stage3

user_query = "What is happening in Nepal Right Now?"

stage1_output = stage1(user_query)
stage2_output = stage2(stage1_output, top_k=3)
final_answer = stage3(live_context, archival_context)
```

## Limitations and Future Work

The system has clear bottlenecks. Web scraping is fragile and depends on site structure. FAISS indexing happens synchronously due to time contraints, blocking the pipeline. The synthesis stage doesn't yet implement source attribution in its output which will involve playing with citation. 

We are still updating our work, open to suggestions!

