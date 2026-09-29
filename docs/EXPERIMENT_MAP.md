# CMC 87708 final experiment map

This map links the revised manuscript analyses to the executable code retained in this repository.

## Common evaluation corpus

- 3 public NetFlow benchmarks
- 500 held-out target edges per dataset
- target allocation proportional to eligible test-window edge counts
- sampling seed 7
- identical target IDs reused across analyses
- ground-truth labels excluded from LLM-facing evidence

## Primary Detector-state comparison (Tables 8-9)

- Models: Gemma-2-9B-IT, Qwen2.5-7B-Instruct
- Evidence: identical complete Detector-state packets
- Policies: Basic vs Source-gated
- Generation: deterministic decoding, max_new_tokens=768
- Native outputs: 6,000 total
- Exact protected fields: prediction, score, threshold, score margin
- Semantic consistency: AlignScore
- Existing code:
  - `notebooks/CMC_R2_Full768_Gemma_Qwen_AlignScore_Colab.ipynb`
  - `scripts/02_run_primary_generation.py`
  - `scripts/03_recompute_primary_metrics.py`
  - `scripts/04_primary_paired_statistics.py`
  - `scripts/05_primary_alignscore.py`

## Gemma four-condition exposure characterization (Section 6.1)

- Model: Gemma-2-9B-IT
- Conditions: LLM-only control, Flow metadata, Graph context, Detector-state condition
- 500 targets × 3 datasets × 4 conditions = 6,000 native notes
- max_new_tokens=768
- Existing code:
  - `notebooks/CMC_R1_Gemma_ThreeState_768_Colab.ipynb`
  - reused Detector-state 768 outputs from the primary run

## Secondary contextual source-boundary ablation (Tables 10-11)

- Models: Gemma-2-9B-IT, Qwen2.5-7B-Instruct
- Scopes: Flow, Flow+Graph
- Policies: Basic, Bounded
- Detector state and traceability excluded from the LLM-facing context
- 2 models × 3 datasets × 500 targets × 2 scopes × 2 policies = 12,000 native notes
- deterministic decoding, max_new_tokens=1536
- New final code:
  - `configs/contextual_final.yaml`
  - `scripts/06_run_contextual_ablation.py`
  - `scripts/07_evaluate_contextual_ablation.py`
  - `notebooks/CMC_FINAL_Contextual_Gemma_Qwen_1536_Colab.ipynb`

## Statistical interpretation

- Binary paired outcomes: paired risk difference, 95% paired bootstrap CI, exact McNemar test
- Continuous/AlignScore: paired mean/median differences, paired bootstrap CI, Wilcoxon signed-rank test
- 10,000 paired bootstrap resamples
- Holm correction within model-family comparison sets
- Cross-state four-condition rates are descriptive; inferential conclusions are restricted to matched comparisons.
