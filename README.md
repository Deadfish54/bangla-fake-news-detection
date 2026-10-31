# Zero-Shot vs. Fine-Tuned: Bangla Fake News Detection

A cross-dataset study comparing a classical baseline, a fine-tuned Bangla transformer, and open instruction-tuned LLMs (zero-shot and few-shot) for detecting fake news in Bangla.

**Authors:** Mahbubul Alam · Joy Das · Arpita Majumder
**Supervisor:** Antu Chowdhury
East Delta University, Department of Computer Science & Engineering

## Models compared

| Approach | Model |
|---|---|
| Classical baseline | TF-IDF + linear SVM |
| Fine-tuned transformer | BanglaBERT (`csebuetnlp/banglabert`), 3 seeds |
| Open LLMs, zero-shot and 4-shot prompting | Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen2.5-7B-Instruct |

Models are trained (where applicable) on **BanFakeNews-2.0** and evaluated both in-distribution and on **BanglaFakeNews2025**, to measure how well each approach transfers to newer, unseen news.

## Key results (macro-F1, %)

| Model | Setting | BFN-2.0 test | BFN-2025 (cross-dataset) |
|---|---|---|---|
| BanglaBERT | fine-tuned | **92.4 ± 0.3** | 79.4 ± 1.6 |
| TF-IDF + SVM | supervised | 88.2 | **83.2** |

On the shared LLM evaluation subsets:

| Model | Setting | BFN-2.0 subset | BFN-2025 subset |
|---|---|---|---|
| BanglaBERT | fine-tuned | **91.0 ± 0.1** | 80.1 ± 1.5 |
| TF-IDF + SVM | supervised | 86.4 | 84.3 |
| Gemma-2-9B-it | zero-shot | 75.1 | **89.4** |
| Llama-3.1-8B-Instruct | 4-shot | 72.8 ± 1.6 | 86.9 ± 1.7 |
| Qwen2.5-7B-Instruct | 4-shot | 65.3 ± 4.4 | 84.8 ± 0.6 |

**Takeaway:** fine-tuned BanglaBERT is strongest in-distribution but drops about 13 points on newer news, while prompted LLMs generalise better across datasets. Full tables, cross-dataset drop and error analysis are in `04_results.ipynb`.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data_and_svm_baseline.ipynb` | Data loading, cleaning, splits, TF-IDF + SVM baseline |
| `02_banglabert_finetuning.ipynb` | BanglaBERT fine-tuning over 3 seeds |
| `03_llm_prompting_{llama,gemma,qwen}.ipynb` | Zero-shot and few-shot prompting per LLM |
| `04_results.ipynb` | Results tables, cross-dataset drop, error analysis |

## Running

All notebooks were run on Kaggle (GPU). Attach the BanFakeNews-2.0 and BanglaFakeNews2025 datasets, and add a Hugging Face token as a Kaggle secret for the gated LLMs (Llama, Gemma). No tokens are stored in this repo.
