from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, RobertaModel


ALIGNSCORE_PAPER = "Zha et al., ACL 2023"
ALIGNSCORE_BACKBONE = "roberta-large"
ALIGNSCORE_MODE = "nli_sp"


def _safe_sent_tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        out = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        out = [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if s.strip()]
    return out


def _context_chunks(text: str) -> list[str]:
    premise_sents = _safe_sent_tokenize(text) or [""]
    n_chunks_target = len(text.strip().split()) // 350 + 1
    group_size = max(len(premise_sents) // n_chunks_target, 1)
    chunks = [
        " ".join(premise_sents[i : i + group_size])
        for i in range(0, len(premise_sents), group_size)
    ]
    return chunks or [""]


def _fmt_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def packet_to_alignscore_context(packet: dict[str, Any]) -> str:
    """Lossless, method-independent textual serialization of an LLM-facing packet.

    The serialization adds no security interpretation; it only verbalizes field
    names and values so the pretrained text-pair evaluator receives textual
    context rather than raw JSON syntax.
    """
    lines: list[str] = []

    if packet.get("dataset") is not None:
        lines.append(f"Dataset is {_fmt_value(packet['dataset'])}.")
    if packet.get("state") is not None:
        lines.append(f"Evidence state is {_fmt_value(packet['state'])}.")

    target = packet.get("target") or {}
    target_labels = {
        "split": "Data split",
        "window_id": "Window identifier",
        "graph_id": "Graph identifier",
        "edge_id": "Edge identifier",
        "src_ip": "Source IP",
        "dst_ip": "Destination IP",
        "src_port": "Source port",
        "dst_port": "Destination port",
        "protocol": "Protocol",
        "src_node_id": "Source node identifier",
        "dst_node_id": "Destination node identifier",
        "src_endpoint_alias": "Source endpoint alias",
        "dst_endpoint_alias": "Destination endpoint alias",
    }
    for key, label in target_labels.items():
        if key in target:
            lines.append(f"{label} is {_fmt_value(target[key])}.")

    flow = packet.get("flow_evidence") or {}
    for key, label in {
        "src_ip": "Source IP",
        "dst_ip": "Destination IP",
        "src_port": "Source port",
        "dst_port": "Destination port",
        "protocol": "Protocol",
    }.items():
        if key in flow and key not in target:
            lines.append(f"{label} is {_fmt_value(flow[key])}.")

    graph = packet.get("graph_evidence") or {}
    graph_labels = {
        "src_in_degree": "Source in-degree",
        "src_out_degree": "Source out-degree",
        "dst_in_degree": "Destination in-degree",
        "dst_out_degree": "Destination out-degree",
        "repeated_edge_count": "Repeated edge count",
        "src_unique_neighbors": "Source unique-neighbor count",
        "dst_unique_neighbors": "Destination unique-neighbor count",
    }
    for key, label in graph_labels.items():
        if key in graph:
            lines.append(f"{label} is {_fmt_value(graph[key])}.")

    detector = packet.get("detector_state") or {}
    detector_labels = {
        "model_name": "Detector model name",
        "prediction": "Detector prediction",
        "score": "Detector score",
        "threshold": "Decision threshold",
        "score_margin": "Detector score margin",
    }
    for key, label in detector_labels.items():
        if key in detector:
            lines.append(f"{label} is {_fmt_value(detector[key])}.")

    trace = packet.get("traceability") or {}
    trace_labels = {
        "split": "Traceability split",
        "window_id": "Traceability window identifier",
        "graph_id": "Traceability graph identifier",
        "edge_id": "Traceability edge identifier",
        "endpoint_mapping": "Traceability endpoint mapping",
    }
    for key, label in trace_labels.items():
        if key in trace:
            if key in {"split", "window_id", "graph_id", "edge_id"} and key in target:
                continue
            lines.append(f"{label} is {_fmt_value(trace[key])}.")

    return " ".join(lines)


def context_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _AlignScoreNLIModel(nn.Module):
    """Inference-only subset of the official AlignScore-large architecture."""

    def __init__(self, backbone: str = ALIGNSCORE_BACKBONE):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(backbone)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        return self.tri_layer(self.dropout(out.pooler_output))


class AlignScoreLargeNLI:
    """Compatibility scorer for the official AlignScore-large nli_sp metric.

    It uses the official yzha/AlignScore checkpoint and the published nli_sp
    aggregation, but avoids the original package's legacy torch<2 dependency
    pin. No model is trained or fine-tuned here.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cuda",
        batch_size: int = 32,
        backbone: str = ALIGNSCORE_BACKBONE,
    ):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = _AlignScoreNLIModel(backbone=backbone)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        raw_state = ckpt.get("state_dict", ckpt)
        selected = {}
        for key, value in raw_state.items():
            k = key
            for prefix in ("module.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            if k.startswith("base_model.") or k.startswith("tri_layer."):
                selected[k] = value

        required = {"tri_layer.weight", "tri_layer.bias"}
        missing_required = sorted(required - set(selected))
        if missing_required:
            raise RuntimeError(
                "AlignScore checkpoint does not expose expected NLI-head weights: "
                + ", ".join(missing_required)
            )

        incompat = self.model.load_state_dict(selected, strict=False)
        encoder_missing = [k for k in incompat.missing_keys if k.startswith("base_model.")]
        if encoder_missing:
            raise RuntimeError(
                f"AlignScore checkpoint missing {len(encoder_missing)} RoBERTa encoder weights; "
                f"first few: {encoder_missing[:5]}"
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, contexts: list[str], claims: list[str]) -> list[float]:
        if len(contexts) != len(claims):
            raise ValueError("contexts and claims must have the same length")

        pair_pre: list[str] = []
        pair_hyp: list[str] = []
        mapping: list[tuple[int, int]] = []
        sentence_counts: list[int] = []

        for example_idx, (context, claim) in enumerate(zip(contexts, claims)):
            chunks = _context_chunks(context)
            claim_sents = _safe_sent_tokenize(claim)
            if not claim_sents:
                sentence_counts.append(0)
                continue

            sentence_counts.append(len(claim_sents))
            for chunk in chunks:
                for sent_idx, sent in enumerate(claim_sents):
                    pair_pre.append(chunk)
                    pair_hyp.append(sent)
                    mapping.append((example_idx, sent_idx))

        sent_scores: list[list[float]] = [
            [-np.inf] * n for n in sentence_counts
        ]

        for start in range(0, len(pair_pre), self.batch_size):
            end = start + self.batch_size
            enc = self.tokenizer(
                pair_pre[start:end],
                pair_hyp[start:end],
                truncation="only_first",
                padding=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(enc)
            probs = torch.softmax(logits, dim=-1)[:, 0].detach().cpu().numpy()

            for offset, prob in enumerate(probs):
                ex_idx, sent_idx = mapping[start + offset]
                if prob > sent_scores[ex_idx][sent_idx]:
                    sent_scores[ex_idx][sent_idx] = float(prob)

        out: list[float] = []
        for scores in sent_scores:
            if not scores:
                out.append(float("nan"))
            else:
                finite = [s for s in scores if np.isfinite(s)]
                out.append(float(np.mean(finite)) if finite else float("nan"))
        return out
