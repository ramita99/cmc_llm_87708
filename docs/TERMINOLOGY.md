# CMC 87708 final terminology map

This file maps historical implementation identifiers to the terminology used in the revised manuscript.

| Historical/internal identifier | Revised manuscript term | Meaning |
|---|---|---|
| `llm_only` | LLM-only control | Case anchor / evidence-limit control |
| `flow` | Flow metadata | Target-flow metadata only |
| `graph` | Graph context | Target-flow metadata + graph-local context |
| `detector` | Detector-state condition | Target-flow metadata + graph-local context + detector state |
| `basic_prompt` | Basic | Common reporting task without the Source-gated constraint |
| `proposed_source_gated` | Source-gated | Same complete packet + explicit source/protected-field policy |
| `deterministic_template` | deterministic finalization/reference | Safeguard/reference; not counted as native LLM success |
| `flow_evidence` | target-flow metadata | LLM-facing flow facts |
| `graph_evidence` | graph-local context | Seven local structural descriptors |
| `detector_state` | detector state | prediction, score, threshold, score margin |
| `traceability` | traceability metadata | dataset/split/window/edge provenance for audit linkage |

For the secondary contextual ablation, the manuscript terms are used directly:
- scope `flow` = Flow
- scope `flow_graph` = Flow+Graph
- policy `basic` = Basic
- policy `bounded` = Bounded

The contextual ablation intentionally excludes detector state and traceability from the LLM-facing context.
