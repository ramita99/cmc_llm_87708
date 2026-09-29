# CMC 87708 reproducibility code

Public reproducibility package for the revised manuscript:

**A Source-Gated, Detector-Anchored LLM Framework for NetFlow Intrusion Triage with an Edge-Centric GNN Detector**

The upstream edge-centric GNN detector is treated as fixed. This repository reproduces the downstream LLM reporting experiments: matched target construction, Source-gated / Basic generation, protected detector-state auditing, deterministic finalization, the Gemma four-condition characterization, the Gemma/Qwen contextual source-boundary ablation, AlignScore, and paired statistical analysis.

## Final revised-manuscript design

### Primary Detector-state comparison — Tables 8-9
- 3 public NetFlow benchmarks
- 500 held-out target edges per dataset
- Gemma-2-9B-IT and Qwen2.5-7B-Instruct
- Basic vs Source-gated on **identical complete Detector-state packets**
- deterministic decoding, `max_new_tokens=768`
- 6,000 native notes total
- protected fields: prediction, score, threshold, score margin
- original detector record retained independently for audit/finalization
- AlignScore-large, `nli_sp`, as a complementary semantic-consistency measure

### Gemma four-condition characterization — Section 6.1
- LLM-only control
- Flow metadata
- Graph context
- Detector-state condition
- 500 targets × 3 datasets × 4 conditions = 6,000 Gemma notes
- deterministic decoding, `max_new_tokens=768`
- descriptive role-matched characterization; **not** a common performance ladder

### Secondary contextual source-boundary ablation — Tables 10-11
- Gemma-2-9B-IT and Qwen2.5-7B-Instruct
- Flow and Flow+Graph scopes
- Basic and Bounded policies
- detector state and traceability excluded from the LLM-facing contextual input
- 2 models × 3 datasets × 500 targets × 2 scopes × 2 policies = **12,000 native notes**
- deterministic decoding, `max_new_tokens=1536`
- paired inference only within the same model/dataset/scope/target context
- contextual AlignScore uses only the LLM-facing Flow or Flow+Graph context

## Repository layout

```text
configs/
  primary_final.yaml
  contextual_final.yaml
  datasets.example.yaml
docs/
  EXPERIMENT_MAP.md
  TERMINOLOGY.md
notebooks/
  CMC_FINAL_Primary_Gemma_Qwen_768_Colab.ipynb
  CMC_FINAL_FourState_Gemma_768_Colab.ipynb
  CMC_FINAL_Contextual_Gemma_Qwen_1536_Colab.ipynb
scripts/
  01_prepare_packets.py
  02_run_primary_generation.py
  03_recompute_primary_metrics.py
  04_primary_paired_statistics.py
  05_primary_alignscore.py
  06_run_contextual_ablation.py
  07_evaluate_contextual_ablation.py
  08_contextual_alignscore.py
src/cmc87708/
  protocol.py
  sampling.py
  evidence.py
  prompts.py
  evaluation.py
  evaluation_v2.py
  alignscore_compat.py
```

## Data interface

Raw benchmark traffic and fixed detector-output artifacts are not redistributed. Copy `configs/datasets.example.yaml` and fill in the local/Google Drive paths.

Packet construction removes ground-truth fields before target sampling and excludes ground truth from every LLM-facing object. The same 500 held-out target IDs per dataset are reused across paired comparisons.

## Reproduction order

```bash
pip install -e .

# 1. Configure dataset / detector artifact paths.
cp configs/datasets.example.yaml configs/datasets.local.yaml

# 2. Build the matched 1,500-target / four-condition packet corpus.
python scripts/01_prepare_packets.py \
  --experiment configs/primary_final.yaml \
  --datasets configs/datasets.local.yaml

# 3. Primary generation is normally run from the Colab notebook for each model.
# See notebooks/CMC_FINAL_Primary_Gemma_Qwen_768_Colab.ipynb

# 4. Recompute primary exact metrics / paired statistics.
python scripts/03_recompute_primary_metrics.py --experiment <run-config.yaml>
python scripts/04_primary_paired_statistics.py --experiment <run-config.yaml>

# 5. Run the contextual ablation from its Colab notebook, or directly:
python scripts/06_run_contextual_ablation.py --config configs/contextual_final.yaml --model-key gemma --resume
python scripts/06_run_contextual_ablation.py --config configs/contextual_final.yaml --model-key qwen --resume
python scripts/07_evaluate_contextual_ablation.py --config configs/contextual_final.yaml
```

AlignScore requires the official `AlignScore-large.ckpt` checkpoint and is run with the provided primary/contextual AlignScore scripts.

## Statistical analysis

- binary paired outcomes: paired risk difference, 95% paired bootstrap CI, exact McNemar test
- AlignScore / continuous paired outcomes: paired mean/median differences, paired bootstrap CI, Wilcoxon signed-rank test
- 10,000 paired bootstrap resamples
- Holm correction within model-family comparison sets

The four evidence-exposure conditions are not used as a common inferential ranking. Inferential claims are restricted to matched comparisons with identical LLM-facing evidence/context.

## Terminology

Historical implementation IDs are retained where required to reproduce the original artifacts, but `docs/TERMINOLOGY.md` maps them to the terminology used in the revised manuscript.

## Reproducibility scope

The repository reproduces the downstream reporting experiments. It does not claim to reproduce a new detector architecture or benchmark detector accuracy, precision, recall, F1, or calibration.
