# Final validation checklist

Before citing this repository as the reproducibility package for manuscript 87708, run the three public Colab notebooks and verify the following against the revised manuscript.

## Common corpus
- [ ] 500 held-out targets per dataset (1,500 total)
- [ ] sampling seed = 7
- [ ] no ground-truth field in any LLM-facing packet/context
- [ ] identical target IDs reused across all paired analyses

## Primary Detector-state comparison
- [ ] 3 datasets × 500 targets × 2 policies × 2 models = 6,000 native outputs
- [ ] max_new_tokens = 768
- [ ] Basic and Source-gated packet hashes match within every pair
- [ ] Table 8 protected-field rates match the manuscript
- [ ] Table 9 AlignScore means / paired statistics match the manuscript
- [ ] native failures remain counted as LLM failures
- [ ] deterministic finalization is reported separately from native success

## Gemma four-condition characterization
- [ ] 500 × 3 datasets × 4 conditions = 6,000 notes
- [ ] same target IDs as the primary experiment
- [ ] max_new_tokens = 768
- [ ] condition-level descriptive checks match Section 6.1
- [ ] no cross-condition inferential ranking is produced

## Contextual source-boundary ablation
- [ ] Gemma: 6,000 outputs
- [ ] Qwen: 6,000 outputs
- [ ] total = 12,000 native contextual outputs
- [ ] scopes = Flow / Flow+Graph
- [ ] policies = Basic / Bounded
- [ ] max_new_tokens = 1536
- [ ] detector state and traceability excluded from LLM-facing context
- [ ] Basic/Bounded context hashes match within every paired target
- [ ] Table 10 AlignScore values / paired intervals match the manuscript
- [ ] Table 11 unsupported-security-extrapolation rates match the manuscript
- [ ] all six Flow+Graph paired 95% bootstrap intervals reproduce the reported direction

## Model revisions
- [ ] Gemma-2-9B-IT revision = 11c9b309abf73637e4b6f9a3fa1e92e615547819
- [ ] Qwen2.5-7B-Instruct revision = a09a35458c702b33eeacc393d103063234e8bc28

## Release
Only after all checks pass:
1. record the final repository commit SHA in the experiment log;
2. tag the public repository (for example, `cmc-87708-r1-repro`);
3. do not modify prompt wording, packet construction, audit rules, or statistics without creating a new protocol version.
