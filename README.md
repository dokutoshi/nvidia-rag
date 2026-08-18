# Conversational RAG over NVIDIA Documentation

A retrieval-augmented generation system built over the full `docs.nvidia.com` corpus
(~59,000 pages), indexed four different ways to compare embedding backends and chunk sizes,
and served through a conversational retrieval chain with two interchangeable generators.

Built as the NLP mini-project for the Interview Kickstart ML SwitchUp program, Oct–Nov 2023.

---

## Why this project is interesting

Most RAG demos index a handful of PDFs and stop. Four things here go further:

1. **The same pipeline runs on three accelerators.** It was ported across an EC2 GPU instance,
   an M1 MacBook, and Google Colab — each forcing a different quantization strategy to fit
   Llama-2-7B in memory. See *Portability* below; this is the part worth reading first.
2. **The vendor is removable.** One variant runs entirely on HuggingFace embeddings and a local
   quantized Llama-2, with no OpenAI API in the loop at all.
3. **It is a controlled comparison, not one pipeline.** The same corpus was indexed across
   multiple embedding models × chunk sizes × vector stores, holding everything else fixed —
   so differences are attributable rather than anecdotal.
4. **The comparison is measured, not asserted.** Configurations are scored with RAGAS across
   five metrics, with the judge model self-hosted rather than delegated to OpenAI.

---

## Portability across three environments

Llama-2-7B in fp16 needs roughly 13 GB of accelerator memory — more than any of these machines
had spare. Each environment required a different way of not paying that cost:

| Environment | Accelerator | Quantization | Embeddings |
|---|---|---|---|
| **EC2** (NVIDIA NGC container) | CUDA | `BitsAndBytesConfig`, 8-bit, `device_map="auto"` | OpenAI + HuggingFace |
| **M1 MacBook** | Metal / MPS | `LlamaCpp` + GGUF, `n_gpu_layers` offload | OpenAI + HuggingFace |
| **Google Colab** | Free-tier GPU | `bitsandbytes` + `accelerate` | **HuggingFace only** |

The M1 path exists because `bitsandbytes` is CUDA-only and will not run on Apple Silicon at all
— so hitting the same memory target required a completely different toolchain, `llama.cpp` with
GGUF-quantized weights over Metal.

Two notebooks in this repo correspond to these:

- **`NVIDIA_LLM.ipynb`** — 177 cells, executed with outputs retained. Carries both the CUDA and
  Metal paths, since it was ported back and forth between EC2 and the M1. This is the evidence.
- **`NVIDIA_LLM_colab_local.ipynb`** — 64 cells, the Colab variant. No OpenAI dependency
  whatsoever. Adds HTML nav/header stripping before indexing, a refactored sitemap loader
  (`nest_asyncio`), a custom `EOS(StoppingCriteria)`, and MarkupLM as live code.

They share only 25 cells — these are genuinely different builds, not drafts of one another.

### What is varied

| Dimension | Options compared |
|---|---|
| Vector store | Chroma · FAISS *(near-equal reference counts — a genuine head-to-head)* |
| Embeddings | `text-embedding-ada-002` · `all-mpnet-base-v2` · `AzureOpenAIEmbeddings` |
| Chunk size | 100 · 600 *(overlap fixed at 20)* |
| Splitter | `RecursiveCharacterTextSplitter` · `CharacterTextSplitter` |
| Tokenizer | `tiktoken` (OpenAI BPE) · `AutoTokenizer` (Llama) |
| Generator | `Llama-2-7b-chat-hf` (quantized) · `gpt-3.5-turbo` |
| Chain | `ConversationalRetrievalChain` · `RetrievalQA` |
| Retrieval | `similarity_search` · `as_retriever(search_kwargs={"k": 6})` |

---

## Architecture

```
docs.nvidia.com sitemap index (NVIDIA.xml)
    │  3 sitemaps, 59,016 URLs enumerated
    ▼
SitemapLoader  (filter_urls, continue_on_failure=True)
    │  failures logged per-URL → error_log.txt
    ▼
RecursiveCharacterTextSplitter   chunk_size ∈ {100, 600}, chunk_overlap = 20
    │
    ├── text-embedding-ada-002              (OpenAI, 1536-dim)
    └── sentence-transformers/all-mpnet-base-v2   (local, 768-dim)
    │
    ├── Chroma   → chroma_db_{hf4,open_ai}_{100,600}/
    └── FAISS    → faiss_{hf,openAI}_index_{100,600}/
    │
    ▼
ConversationalRetrievalChain   search_kwargs={"k": 6}, langchain.memory
    │
    ├── meta-llama/Llama-2-7b-chat-hf   (local, quantized — see below)
    └── gpt-3.5-turbo                   (API)
    │
    ▼
streaming response  (StreamingStdOutCallbackHandler)
    │
    ▼
RAGAS evaluation   context_precision · context_recall · faithfulness
                   answer_relevancy · harmfulness      (judge: self-hosted LLM)
```

A `RetrievalQA` chain is also present as a single-turn baseline against the conversational one.
Tokenization differs per path: `tiktoken` for the OpenAI models, `AutoTokenizer.from_pretrained(model_id)`
for Llama-2.

---

## Quantization detail

The exact CUDA configuration, for reference:

```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit_fp32_cpu_offload=False,
    llm_int8_threshold=200.0,
)
```

with `device_map="auto"` and `AutoTokenizer.from_pretrained(model_id)`. The Metal path uses
`LlamaCpp(model_path=..., n_gpu_layers=..., n_ctx=..., n_batch=...)` against GGUF weights.

> **Scope note:** this is quantized **inference**. There is no LoRA/PEFT adapter *training*
> anywhere in either notebook — `LoraConfig`, `lora_alpha`, `get_peft_model`, and `SFTTrainer`
> are all absent across all 177 and 64 cells. The work is accurately described as 8-bit and
> GGUF quantization to fit a 7B model under a memory ceiling, not as QLoRA fine-tuning.

---

## The index comparison

Seven of eight possible cells were built. Sizes are the built artifacts on disk:

| Store | Embeddings | Chunk | Size |
|---|---|---:|---:|
| `chroma_db_open_ai_100` | ada-002 | 100 | 17.9 GiB |
| `chroma_db_hf4_100` | all-mpnet-base-v2 | 100 | 7.4 GiB |
| `faiss_hf_index_100` | all-mpnet-base-v2 | 100 | 2.7 GiB |
| `chroma_db_open_ai_600` | ada-002 | 600 | 2.4 GiB |
| `chroma_db_hf4_600` | all-mpnet-base-v2 | 600 | 1.0 GiB |
| `faiss_openAI_index_600` | ada-002 | 600 | 868 MiB |
| `faiss_hf_index_600` | all-mpnet-base-v2 | 600 | 470 MiB |
| *`faiss_openAI_index_100`* | *ada-002* | *100* | *not built* |

**What the numbers show:**

- **Chunk size dominates everything.** Going from 100 → 600 shrinks each store by ~6–7×
  consistently across both engines and both backends. With overlap fixed at 20, larger chunks
  mean far fewer vectors — the single biggest lever on index cost.
- **Embedding dimensionality costs about what you'd expect.** At equal chunk size the OpenAI
  stores run ~2.4× the HuggingFace ones, tracking the 1536 vs 768 dimension ratio plus
  Chroma's parquet overhead.
- **FAISS is dramatically leaner than Chroma** for the same vectors — compare
  `faiss_hf_index_100` (2.7 GiB) against `chroma_db_hf4_100` (7.4 GiB), a ~2.7× difference,
  since Chroma persists both a parquet embedding table and its own HNSW index.
- **The two 600-chunk FAISS stores share a byte-identical docstore** (`index.pkl`, 74,198,719 B),
  differing only in `index.faiss` — confirming the comparison held chunking genuinely fixed
  and varied only the embedding model.

The missing cell is the one that would have been most expensive: OpenAI embeddings at
chunk_size=100 is the largest configuration, and its Chroma equivalent alone is 17.9 GiB.

---

## Evaluation

Configurations are scored with [RAGAS](https://github.com/explodinggradients/ragas) rather than
judged by inspection. Five metrics, covering both halves of the pipeline:

| Metric | What it measures |
|---|---|
| `context_precision` | Are the retrieved chunks actually relevant? |
| `context_recall` | Did retrieval find everything it needed? |
| `faithfulness` | Is the answer grounded in the retrieved context, or hallucinated? |
| `answer_relevancy` | Does the answer address the question asked? |
| `harmfulness` | Critique-based safety check on generated output |

The first two isolate **retrieval** quality; the next two isolate **generation** quality. That
separation is what makes the vector-store and chunk-size comparison meaningful — a bad answer
can be attributed to the retriever or the generator rather than to the system as a whole.

**The judge is self-hosted.** RAGAS defaults to calling OpenAI for every metric evaluation,
which is both a cost and a data-egress decision. Here each metric's evaluator is reassigned to
a locally-served model through the `LangchainLLM` wrapper:

```python
from ragas.llms import LangchainLLM
from ragas.metrics import context_precision, answer_relevancy, faithfulness, context_recall
from ragas.metrics.critique import harmfulness

faithfulness.llm     = vllm
answer_relevancy.llm = vllm
context_precision.llm = vllm
context_recall.llm   = vllm
harmfulness.llm      = vllm
```

Evaluating a 59k-page corpus with an API-based judge across several configurations would have
been prohibitively expensive; swapping the judge makes the comparison affordable to run
repeatedly.

### Two evaluation layers — one ran, one was blocked

**Qualitative comparison (ran).** Section *"Competitive comparisons of LLM and Retrievers"*
(cells 106–140) executed and its outputs are saved in the notebook. It runs the same questions
across the retriever × generator grid — Chroma retrievers and FAISS retrievers, against both
GPT-3.5 and Llama-2 — so the generated answers sit side by side for direct inspection. Of 120
code cells, 51 carry saved output.

**Quantitative scoring (blocked).** The RAGAS section (cell 141 onward) did not complete. It
failed at import:

```
ImportError: cannot import name 'AzureOpenAIEmbeddings' from 'langchain.embeddings'
```

This is library drift, not a design gap, and it was diagnosed at the time — the section's own
note reads *"this requires OpenAIEmbeddings in the langchain package. Currently there are
issues using..."*. A related break appears earlier in the notebook: five cells fail with
`AttributeError: module 'openai' has no attribute 'error'`, the openai SDK v1.0 removal of the
`openai.error` namespace. Seven cells error in total; both causes are version incompatibilities
introduced by upstream releases in late 2023.

**Restoring it is small — two one-line changes.**

The `openai.error` failures have a direct cause visible in the notebook itself:

```python
%pip install openai #==0.28.1
```

The version pin is commented out. Installing unpinned in late 2023 pulled openai v1.x, which
removed the `openai.error` namespace — uncommenting the pin resolves all five failures.

For the RAGAS import: `AzureOpenAIEmbeddings` moved to the `langchain_openai` package in the
LangChain split. Neither notebook imports `langchain_openai` or `langchain_community`, which
dates the stack to pre-split LangChain 0.0.x — useful to know for anyone reproducing this.

The vector stores already exist and the judge is self-hosted, so a rerun is compute-only with
no paid API spend.

That rerun is the highest-value remaining work here: it converts the central claim — that
these configurations differ measurably — from side-by-side reading into a scored table of five
metrics across the seven built index configurations. [^1]

---

## Repository layout

```
NVIDIA_LLM.ipynb              EC2 ↔ M1 build — 177 cells, executed, outputs retained.
                              Both CUDA and Metal quantization paths; RAGAS section.
NVIDIA_LLM_colab_local.ipynb  Colab build — 64 cells, no OpenAI dependency.
                              HTML chrome stripping, refactored loader, custom stopping.
test.ipynb                    scratch / verification
NVIDIA.xml                    sitemap index defining the corpus
error_log.txt                 crawl log; per-URL failures + the 59,016-page progress record
manifest.json                 full inventory: sizes, configs, provenance, open questions
README.md                     this file
```

The two notebooks share only 25 cells. Keep both — one is the executed evidence across two
accelerators, the other is the vendor-free variant.

Vector stores are **not** in version control — see below.

---

## Credentials

Both notebooks read secrets from a `.env` file via `python-dotenv` — nothing is hardcoded:

```bash
cp .env.example .env      # then fill in your own values
```

| Variable | Needed for |
|---|---|
| `HF_AUTH_TOKEN` | Gated model download — `meta-llama/Llama-2-7b-chat-hf` |
| `OPENAI_API_KEY` | The ada-002 embedding and `gpt-3.5-turbo` paths only |

`NVIDIA_LLM_colab_local.ipynb` needs only `HF_AUTH_TOKEN` — it has no OpenAI dependency at all.

> The 2023 notebooks originally carried these values inline. They have been replaced with
> environment lookups and the original credentials revoked.

---

## Reproducing

1. Crawl: load `NVIDIA.xml` through LangChain's `SitemapLoader` with `filter_urls`
   and `continue_on_failure=True`.
2. Split: `RecursiveCharacterTextSplitter(chunk_size=100|600, chunk_overlap=20)`.
3. Embed: `text-embedding-ada-002` or `sentence-transformers/all-mpnet-base-v2`.
4. Persist to Chroma or FAISS under the matching directory name.
5. Query: `ConversationalRetrievalChain` with `search_kwargs={"k": 6}`.

**Two warnings before rerunning:**

- **Cost.** Re-embedding ~59k pages at `chunk_size=100` with ada-002 is a substantial paid
  API run. The HuggingFace configurations are compute-only and free.
- **Drift.** `docs.nvidia.com` has changed considerably since the October 2023 crawl. A fresh
  crawl will *not* reproduce these indexes. That irreproducibility is the main reason the
  artifacts are preserved rather than regenerated on demand.

---

## Artifacts and storage

The built indexes total **32.7 GiB** against **13 MB** of source — 99.96% of the project by
size is derived data. They are tracked with DVC rather than git, on an S3 remote.

On APFS, set the DVC cache to use copy-on-write reflinks before adding, or `dvc add` will
duplicate all 32.7 GiB on local disk:

```bash
dvc config cache.type reflink,hardlink,symlink,copy
dvc pull   # retrieve the indexes
```

---

## Environment

Development spanned three environments, all visible in the source: CUDA device selection inside
an NVIDIA NGC TensorFlow container (`/opt/tensorflow/.../python3.10`) for EC2, the MPS and
`llama.cpp` paths for the M1, and `from google.colab import drive` in the Colab variant.

Core stack: LangChain (pre-split 0.0.x) · transformers · torch · sentence-transformers ·
chromadb · faiss · llama-cpp-python · bitsandbytes · accelerate · openai · tiktoken · ragas

---

## Known gaps

- Post-split chunk counts are recoverable from notebook outputs but not yet extracted here.
- Which generator produced the final presented results is not recorded.
- `nlp_complete.zip` (~24.5 GiB, packaged 2024-03-28) has not been verified against the
  live tree.

---

[^1]: A note on the timing, since it explains most of the breakage. I built the OpenAI indexes
during the week of 20 November 2023 — which turned out to be the week OpenAI's board fired Sam
Altman, Microsoft moved to hire him, and he was back as CEO a few days later. Everyone pivoted
to Azure-hosted OpenAI overnight, and `AzureOpenAIEmbeddings` was being shuffled between
LangChain packages while I was trying to import it. The `openai` SDK v1.0 rewrite had landed a
couple of weeks before that and removed the `openai.error` namespace, which accounts for the
rest of the failed cells. A weird week to be building on that stack.
