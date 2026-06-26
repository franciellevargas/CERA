<div align="center">

<h1>CERA</h1>

<h3>Beyond Topical Similarity: Contrastive Evidence Retrieval with Interpretable Attention Alignment in RAG</h3>

<p>
  <a href="https://arxiv.org/abs/2606.01482"><img src="https://img.shields.io/badge/arXiv-2606.01482-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.5281/zenodo.17619083"><img src="https://zenodo.org/badge/1079456425.svg" alt="DOI"></a>
</p>

</div>

This repository contains the code accompanying our preprint
**[Beyond Topical Similarity: Contrastive Evidence Retrieval with Interpretable Attention Alignment in RAG](https://arxiv.org/abs/2606.01482)**.

CERA fine-tunes a dense retriever (Contriever) to go *beyond topical similarity* by
(1) contrasting evidence-bearing passages against merely on-topic ones and
(2) aligning the model's token-level attention with human-annotated rationales, so
that retrieval becomes both more accurate and interpretable in a RAG setting.

## Requirements

- **Python ≥ 3.9** (developed and tested with Python 3.12)
- Install the third-party packages:

  ```bash
  pip install torch transformers numpy pandas tqdm spacy textblob openai mistralai
  ```

- The spaCy / TextBlob scripts additionally need the language model and corpora:

  ```bash
  python -m spacy download en_core_web_sm
  python -m textblob.download_corpora
  ```

- A CUDA-capable GPU is recommended for fine-tuning and evaluation (the code falls
  back to CPU automatically). `openai` and `mistralai` are only required for the
  LLM-as-a-judge evaluation, which expects valid API keys.

## Repository structure

```
CERA/
├── README.md
└── code/
    ├── generate_triplets_train.py          # Triplet construction (shared helpers)
    ├── generate_triplets_nsubj_train.py    # Subjectivity-filtered triplets
    ├── generate_triplets_ev_offset_nsubj.py# Triplets with evidence offsets
    ├── weighted_masking.py                  # POS-weighted token masks
    ├── finetune_triplet_kl_sched.py         # Fine-tuning: triplet + KL alignment
    ├── finetune_weighted_kl.py              # Fine-tuning: triplet + weighted KL
    ├── format_test_data.py                  # Build grouped retrieval test sets
    ├── eval_local.py                        # Retrieval evaluation (Recall@k, etc.)
    ├── generate_rationales_token_level.py   # Token-level rationale extraction
    ├── eval_faithfulness.py                 # Faithfulness evaluation
    ├── eval_llm_judge.py                     # LLM-as-a-judge evaluation
    ├── utils.py                             # Shared helpers (e.g. GPU selection)
    └── results/                             # Stored LLM-judge judgments & summaries
```

### File purposes

**Data / triplet generation**
- [`generate_triplets_train.py`](code/generate_triplets_train.py) — Builds query/positive/negative triplets from the prompt and annotation CSVs and the source documents. Defines the shared helpers (`build_query`, `chunk_text_with_offsets`, `overlaps`) reused by the other generators.
- [`generate_triplets_nsubj_train.py`](code/generate_triplets_nsubj_train.py) — Variant that additionally scores chunk *subjectivity* (TextBlob) to produce the `SUBJ_LOCAL` training triplets.
- [`generate_triplets_ev_offset_nsubj.py`](code/generate_triplets_ev_offset_nsubj.py) — Generates triplets that keep the character offsets of the gold evidence span, used by the weighted-masking / KL-alignment objective.

**Attention alignment**
- [`weighted_masking.py`](code/weighted_masking.py) — Computes POS-weighted token masks (spaCy) over the evidence spans, providing the supervision target for the attention-alignment (KL) loss.

**Fine-tuning**
- [`finetune_triplet_kl_sched.py`](code/finetune_triplet_kl_sched.py) — Fine-tunes Contriever with a contrastive triplet loss plus a scheduled KL term that aligns token attention with the rationale masks.
- [`finetune_weighted_kl.py`](code/finetune_weighted_kl.py) — Fine-tuning variant using the POS-weighted KL alignment objective.

**Evaluation**
- [`format_test_data.py`](code/format_test_data.py) — Groups the test triplets by `(PMCID, query)` into the local/global retrieval evaluation sets.
- [`eval_local.py`](code/eval_local.py) — Runs local (per-document) retrieval evaluation, reporting Recall@k and related metrics.
- [`generate_rationales_token_level.py`](code/generate_rationales_token_level.py) — Extracts token-level relevance/rationale scores from a fine-tuned model (merging WordPiece subtokens back into words).
- [`eval_faithfulness.py`](code/eval_faithfulness.py) — Measures faithfulness by masking the most relevant tokens and observing the effect on retrieval.
- [`eval_llm_judge.py`](code/eval_llm_judge.py) — LLM-as-a-judge evaluation that grades retrieved spans for factual consistency and usefulness against gold spans (OpenAI / Mistral backends).

**Utilities & results**
- [`utils.py`](code/utils.py) — Shared helpers, e.g. `get_free_gpu()` for selecting the least-loaded GPU.
- [`code/results/`](code/results/) — Stored LLM-judge outputs (`*_judgments.json`, `*_summary.json`) for the baseline vs. fine-tuned retriever across judge models (GPT, Qwen3-Max, Mistral-Large) and rank cut-offs.

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{vargas2026topicalsimilaritycontrastiveevidence,
      title={Beyond Topical Similarity: Contrastive Evidence Retrieval with Interpretable Attention Alignment in RAG}, 
      author={Francielle Vargas and João Robiatti and Diego Alves and Lucas Pascotti Valem and Maximilian Seeth and Sebastián Ferrada and Ameeta Agrawal and Daniel Pedronette and André Freitas},
      year={2026},
      eprint={2606.01482},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.01482}, 
}
```
