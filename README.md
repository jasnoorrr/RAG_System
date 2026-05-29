# RAG System — Retrieval-Augmented Generation with Corrective Retrieval

A research-oriented information retrieval pipeline built for CPSC 461 (Applied Machine Learning) at the University of Northern British Columbia. This project investigates how semantically similar distractor documents degrade retrieval quality, and how advanced reranking and corrective retrieval strategies can recover accuracy.

---

## What This Project Does

Given a natural language query, the system retrieves the most relevant documents from an AI-focused news corpus and generates a grounded answer using a language model.

**Core research question:** How do we make retrieval robust when semantically similar but wrong documents compete with the correct answer?

---

## Pipeline Overview

```
Query → Candidate Retrieval → Confidence Estimation → [Corrective Retrieval if needed] → Reranking → Generation
```

---

## Key Results

Evaluated on a held-out test split (11 queries) against 8 retrieval methods:

| Method | Hit@10 | MRR@10 | nDCG@10 | Avg Distractors Above Gold |
|--------|--------|--------|---------|---------------------------|
| Dense (baseline) | 0.5455 | 0.2736 | 0.3370 | 6.00 |
| BM25 | 0.4545 | 0.3485 | 0.3755 | 5.73 |
| TF-IDF | 0.4545 | 0.3485 | 0.3755 | 5.73 |
| Hybrid | 0.4545 | 0.3485 | 0.3755 | 5.73 |
| Logistic Ranker | best per-query rank improvements vs dense | — | — | reduced on specific queries |
| Cross-Encoder | reranked hybrid candidates | — | — | — |
| Defended Dense | confidence-gated hybrid fusion | — | — | — |
| Corrective | query rewriting + cross-encoder reranking | — | — | — |

**Key finding:** The logistic ranker showed per-query rank improvement over dense on specific queries (gold rank 1 on 4/11 test queries vs dense). The cross-encoder confidence estimation correctly classified HIGH confidence retrievals with large score gaps (top gap: 17.0), triggering INITIAL retrieval mode and avoiding unnecessary correction overhead.

---

## System Components

### 1. Corpus Construction
- Filters 209,527 HuffPost news articles to an AI-focused subset using keyword prefiltering + semantic similarity
- Semantic scoring with `all-MiniLM-L6-v2` against AI topic descriptions
- Threshold sensitivity analysis across 0.20–0.35; final threshold 0.25 yielding 65 high-relevance documents

### 2. Auto Query Generation
- Uses `FLAN-T5-small` to generate natural language questions from article text
- Quality filtering pipeline: length check, bad output detection, near-copy detection, question-form validation
- Rule-based fallback generator for edge cases
- 65 query-gold pairs generated; 70/15/15 train/val/test split → 45 train / 9 val / 11 test

### 3. Retrieval Methods (5 Baselines + 3 Advanced)
| Method | Description |
|--------|-------------|
| Dense | `all-MiniLM-L6-v2` embeddings + cosine similarity |
| BM25 | Sparse keyword retrieval (Okapi BM25) |
| TF-IDF | Cosine similarity on TF-IDF vectors |
| Hybrid | Normalized score fusion: α·Dense + (1-α)·BM25 |
| Logistic Ranker | Trained on [dense, BM25, TF-IDF, genericness] features; 915 training instances (45 positive, 870 negative) |
| Cross-Encoder | `ms-marco-MiniLM-L-6-v2` reranks hybrid candidates |
| Defended Dense | Genericness penalty + multi-signal fusion |
| Corrective | Confidence-gated: HIGH → return directly; MEDIUM/LOW → query rewriting + expanded retrieval + reranking |

### 4. Hard Negatives
- Top-ranked semantically similar wrong documents per query
- Used in logistic ranker training to improve discrimination
- Example: query about "AI genius grant" → hard negative was "Ai Weiwei art" (name collision)

### 5. Retrieval Confidence Estimation
- Estimates confidence from top-1 score and score gap between rank-1 and rank-2
- Thresholds tuned on validation split: high_top1=3.58, high_gap=11.87
- Demonstrated on test: gap of 17.02 correctly labeled HIGH, avoiding unnecessary correction

### 6. Corrective Retrieval
- Triggered on MEDIUM/LOW confidence results
- Query rewriting: original + expanded (+ AI/ML terms) + compressed (stopwords removed)
- Merges candidates across variants, reranks with cross-encoder

### 7. Answer Generation
- `FLAN-T5-small` generates grounded answers from top-3 retrieved documents
- Prompt-engineered to avoid hallucination, label leakage, and document reference artifacts
- Example: query "Who won the genius grant?" → correctly retrieved gold doc → generated answer: "ai jen poo"

---

## Ablation Study

Full ablation showing contribution of each pipeline stage:

```
Dense (0.55 Hit@10)
  → Hybrid (0.45 Hit@10, +MRR improvement)
    → Hybrid + Logistic (per-query rank gains)
      → Hybrid + CrossEncoder (reranking quality)
        → CrossEncoder + Confidence + Correction (full system)
```

---

## Honest Limitations

- Small corpus (65 docs) limits generalizability — results on larger corpora would be more robust
- FLAN-T5-small question generation produced many "What is the best title?" queries, weakening evaluation diversity
- Small test set (11 queries) means metric variance is high
- These limitations are documented and acknowledged in the ablation analysis

---

## Tech Stack

- **Python 3.14**
- **SentenceTransformers** — `all-MiniLM-L6-v2` (384-dim embeddings)
- **HuggingFace Transformers** — `FLAN-T5-small` for generation + question generation
- **CrossEncoder** — `ms-marco-MiniLM-L-6-v2`
- **rank-bm25** — Okapi BM25
- **scikit-learn** — TF-IDF, Logistic Regression, cosine similarity
- **NumPy / Pandas / Matplotlib**
- **PyTorch** (CPU inference)

---

## Dataset

[HuffPost News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) — 209,527 articles across 42 categories.

> Download from Kaggle and place `News_Category_Dataset_v3.json` in the project root before running.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/jasnoorrr/RAG_System.git
cd RAG_System

# 2. Install dependencies
pip install sentence-transformers transformers rank-bm25 scikit-learn numpy pandas matplotlib torch

# 3. Download dataset from Kaggle
# Place News_Category_Dataset_v3.json in the project root

# 4. Open and run the notebook
jupyter notebook ragsystem.ipynb
```

---

## Project Context

Built for CPSC 461 — Applied Machine Learning at UNBC (Winter 2026).  
Focus: retrieval robustness under distractor pressure and corrective retrieval strategies.

---

## Team

Built collaboratively by **Jasnoor K. Batra**, **Nainpreet Kaur**, **Sangita Paudel** as part of CPSC 461 — Applied Machine Learning at UNBC (Winter 2026).

[Jasnoor's LinkedIn](https://www.linkedin.com/in/jasnoor-batra) | [Jasnoor's GitHub](https://github.com/jasnoorrr)
