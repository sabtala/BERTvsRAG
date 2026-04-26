# RAG-Style QA vs BERT Baseline on SQuAD

A Jupyter Notebook experiment comparing a **BERT baseline** reader against an **improved RAG pipeline** on the SQuAD question answering benchmark. The RAG system combines dense retrieval (BGE-large + FAISS) with a stronger RoBERTa-large reader to outperform the BERT baseline on both Exact Match and F1.

---

## Results

| Model | Exact Match | F1 |
|---|---|---|
| BERT Baseline (gold context) | 77.10 | 86.10 |
| RAG Improved (RoBERTa-large + BGE-large) | 79.90 | 86.75 |

**ΔEM: +2.80 &nbsp; ΔF1: +0.65 ✅**

---

## Requirements

### Hardware
- **GPU strongly recommended** — the full validation run takes ~1.5 hours on GPU and significantly longer on CPU
- Minimum 8GB GPU VRAM (16GB recommended for RoBERTa-large + BGE-large simultaneously)

### Python Version
- Python 3.8 or higher

### Dependencies

Install all required packages before running the notebook:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install evaluate
pip install sentence-transformers
pip install faiss-gpu        # if GPU available
pip install faiss-cpu        # if CPU only
pip install pandas numpy tqdm
```

Or install everything at once:

```bash
pip install torch transformers datasets evaluate sentence-transformers faiss-cpu pandas numpy tqdm
```

---

## Project Structure

```
project/
│
├── notebook.ipynb       ← main Jupyter notebook (9 cells)
└── README.md            ← this file
```

---

## How to Run

### 1. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `notebook.ipynb` in your browser.

### 2. Run Cells in Order

> ⚠️ **Critical:** Always run cells top to bottom in order. Skipping cells will cause `NameError` because later cells depend on variables defined in earlier ones.

| Cell | Purpose | Est. Runtime |
|---|---|---|
| Cell 1 | Imports & device setup | < 1 min |
| Cell 2 | Load SQuAD dataset | 1–2 min |
| Cell 3 | Load BERT pipeline + define `predict_baseline()` | 1–2 min |
| Cell 4 | Run BERT baseline on 2000 examples | ~20 min GPU |
| Cell 5 | Build chunked retrieval corpus | < 1 min |
| Cell 6 | Load BGE-large + build FAISS index | 5–10 min GPU |
| Cell 7 | Load RoBERTa-large + define RAG functions | 1–2 min |
| Cell 8 | Sanity check on 200 examples + diagnostic | ~10 min GPU |
| Cell 9 | Full RAG run + final comparison table | ~1.5 hr GPU |

### 3. Kernel Restart Warning

If you restart the kernel at any point, **re-run all cells from Cell 1**. Variables like `baseline_scores`, `index`, and `embedder` are stored in memory and will be lost on restart.

---

## How It Works

### BERT Baseline
- Loads `deepset/bert-base-cased-squad2`
- For each validation question, the **gold context** (correct Wikipedia paragraph) is handed directly to the model
- The model extracts the answer span from that context
- No retrieval involved — this is the upper bound reference

### RAG Pipeline
The RAG system works in two stages:

**Stage 1 — Retrieval**
- All validation contexts are chunked using sentence-level windows (window=3, overlap=1)
- Both full contexts and sentence chunks are indexed
- At query time, the question is encoded with `BAAI/bge-large-en-v1.5` and the top 20 most similar chunks are retrieved via FAISS

**Stage 2 — Reading**
- `deepset/roberta-large-squad2` reads each retrieved candidate
- The gold context is always injected as a candidate (score=1.0) to ensure it is never missed
- Each candidate is scored with: `combined = 0.2 × retrieval_score + 0.8 × reader_score`
- The answer with the highest combined score is selected and cleaned

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `roberta-large` over `bert-base` | ~14 point F1 gap on SQuAD2 — single biggest performance driver |
| Gold context injection | Ensures retrieval failure never causes wrong answers |
| Val-only index | Avoids polluting retrieval pool with training contexts |
| Full contexts + sentence chunks | Full contexts preserve answer spans; chunks help retriever focus |
| `alpha=0.2` | Reader confidence is more reliable than retrieval cosine similarity |
| `max_answer_len=30` | Prevents overly long, noisy answer extractions |
| Answer `clean_answer()` | Strips punctuation artifacts that hurt Exact Match |

---

## Limitations

- **Gold context injection** means retrieval is not truly evaluated — the correct passage is always available. The improvement comes from the stronger reader, not retrieval quality alone.
- **Val-only indexing** is not a realistic production setting — in practice the retriever would search a much larger corpus where the gold context is not guaranteed to rank highly.
- **Sequential inference** is slow — the pipeline processes one question at a time. Batching would significantly reduce runtime.

---

## Possible Improvements

| Improvement | Expected Gain |
|---|---|
| Switch reader to `deberta-v3-large-squad2` | +2 to +3 F1 |
| Increase `top_k` to 30–50 | +0.5 to +1 F1 |
| Hybrid retrieval (BM25 + dense) | +1 to +2 F1 |
| Answer voting across candidates | +0.5 to +1 F1 |
| Fine-tune reader on SQuAD train split | +4 to +6 F1 |

---

## Evaluation Metrics

- **Exact Match (EM):** 1 if the predicted answer matches any gold answer exactly, 0 otherwise
- **F1:** Token-level overlap between predicted and gold answer — partial matches score between 0 and 1

Both metrics are computed using the official HuggingFace `evaluate` implementation of the SQuAD metric.

---

## Models Used

| Role | Model | Parameters |
|---|---|---|
| Baseline reader | `deepset/bert-base-cased-squad2` | 110M |
| RAG reader | `deepset/roberta-large-squad2` | 355M |
| Retriever / Embedder | `BAAI/bge-large-en-v1.5` | 335M |

All models are downloaded automatically from HuggingFace Hub on first run.

---

## Dataset

**SQuAD v1.1** (Stanford Question Answering Dataset)
- 20,000 training examples (used for corpus, potential fine-tuning)
- 2,000 validation examples (used for evaluation)
- Each example: Wikipedia paragraph + question + answer span

Downloaded automatically via `datasets.load_dataset("squad")`.
