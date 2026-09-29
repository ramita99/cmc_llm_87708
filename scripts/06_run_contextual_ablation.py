from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_hash(value) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def canonical_target_key(packet: dict) -> tuple[str, str, str]:
    t = packet.get("traceability", {})
    return (
        str(packet.get("dataset")),
        str(t.get("window_id")),
        str(t.get("edge_id")),
    )


def extract_target_flow(packet: dict) -> dict:
    target = packet.get("target", {})
    flow = packet.get("flow_evidence", {})
    fields = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol")
    out = {}
    for name in fields:
        if name in flow:
            out[name] = flow[name]
        elif name in target:
            out[name] = target[name]
    return out


def build_context(packet: dict, scope: str) -> dict:
    context = {"target_flow_metadata": extract_target_flow(packet)}
    if scope == "flow_graph":
        context["graph_local_context"] = dict(packet.get("graph_evidence", {}))
    elif scope != "flow":
        raise ValueError(f"Unknown scope: {scope}")
    # Detector state and traceability are deliberately excluded from LLM-facing context.
    return context


COMMON_TASK = (
    "Write a concise analyst-facing NetFlow triage note using the supplied context. "
    "Report only information that is present in the context."
)

BOUNDED_RULE = (
    "Do not assert an attack type, attacker intent, attribution, causal-security interpretation, "
    "or remediation action unless it is explicitly supported by the supplied context."
)


def prompt_for(context: dict, policy: str) -> str:
    if policy == "basic":
        instruction = COMMON_TASK
    elif policy == "bounded":
        instruction = COMMON_TASK + "\n" + BOUNDED_RULE
    else:
        raise ValueError(policy)
    return instruction + "\n\nAUTHORIZED CONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2)


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def load_model(model_id: str, dtype_name: str, attention_impl: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "dtype": resolve_dtype(dtype_name),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if attention_impl:
        kwargs["attn_implementation"] = attention_impl
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)
    model.eval()
    return model, tokenizer


def _normalize_eos_ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    rendered = [render_chat(tokenizer, p) for p in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    eos_ids = _normalize_eos_ids(model.generation_config.eos_token_id)
    if not eos_ids:
        eos_ids = _normalize_eos_ids(tokenizer.eos_token_id)
    eos_arg = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_arg,
            use_cache=True,
        )

    prefix_len = inputs["input_ids"].shape[-1]
    rows = []
    eos_set = set(eos_ids)
    for seq in output:
        ids = seq[prefix_len:].tolist()
        terminal = None
        for i, token_id in enumerate(ids):
            if token_id in eos_set or token_id == tokenizer.pad_token_id:
                terminal = i
                break
        effective = ids if terminal is None else ids[:terminal]
        rows.append({
            "text": tokenizer.decode(effective, skip_special_tokens=True).strip(),
            "generated_tokens": len(effective),
            "terminated_by_eos": terminal is not None,
            "truncated_at_max_tokens": terminal is None and len(ids) >= max_new_tokens,
        })
    return rows


def batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def prepare_cases(source_root: Path, datasets: list[str], targets_per_dataset: int) -> dict[str, list[dict]]:
    cases = {}
    for dataset in datasets:
        packets = load_jsonl(source_root / "packets" / f"{dataset}.jsonl")
        detector = [p for p in packets if p.get("state") == "detector"]
        if len(detector) != targets_per_dataset:
            raise AssertionError(f"{dataset}: expected {targets_per_dataset} detector packets, got {len(detector)}")
        keys = [canonical_target_key(p) for p in detector]
        if len(set(keys)) != len(keys):
            raise AssertionError(f"{dataset}: duplicate target keys")
        cases[dataset] = detector
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/contextual_final.yaml")
    ap.add_argument("--model-key", choices=["gemma", "qwen"], required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    source_root = Path(cfg["source_packet_root"])
    output_root = Path(cfg["output_root"]) / args.model_key
    datasets = list(cfg["expected_datasets"])
    scopes = list(cfg["scopes"])
    policies = list(cfg["policies"])
    n = int(cfg["targets_per_dataset"])
    batch_size = int(cfg.get("generation_batch_size", 4))
    max_new_tokens = int(cfg["max_new_tokens"])

    spec = cfg["models"][args.model_key]
    model_id = spec["model_id"]
    model_revision = spec.get("revision")
    cases = prepare_cases(source_root, datasets, n)

    model, tokenizer = load_model(
        model_id,
        str(cfg.get("generation_dtype", "bfloat16")),
        str(cfg.get("attention_implementation", "sdpa")),
        model_revision,
    )

    for dataset in datasets:
        for scope in scopes:
            base_cases = cases[dataset]
            contexts = [build_context(p, scope) for p in base_cases]
            context_hashes = [stable_hash(c) for c in contexts]

            for policy in policies:
                out_file = output_root / "outputs" / scope / policy / f"{dataset}.jsonl"
                out_file.parent.mkdir(parents=True, exist_ok=True)

                completed = set()
                if args.resume and out_file.exists():
                    completed = {
                        (str(r["window_id"]), str(r["edge_id"]))
                        for r in load_jsonl(out_file)
                    }

                pending = []
                for packet, context, chash in zip(base_cases, contexts, context_hashes):
                    t = packet["traceability"]
                    key = (str(t.get("window_id")), str(t.get("edge_id")))
                    if key in completed:
                        continue
                    pending.append({
                        "dataset": dataset,
                        "window_id": t.get("window_id"),
                        "edge_id": t.get("edge_id"),
                        "scope": scope,
                        "policy": policy,
                        "context": context,
                        "context_hash": chash,
                        "prompt": prompt_for(context, policy),
                    })

                mode = "a" if args.resume and out_file.exists() else "w"
                with out_file.open(mode, encoding="utf-8") as f:
                    total_batches = (len(pending) + batch_size - 1) // batch_size
                    for batch in tqdm(batches(pending, batch_size), total=total_batches,
                                      desc=f"{args.model_key}:{dataset}:{scope}:{policy}"):
                        gens = generate_batch(model, tokenizer, [r["prompt"] for r in batch], max_new_tokens)
                        for meta, gen in zip(batch, gens):
                            row = {
                                **{k: v for k, v in meta.items() if k != "prompt"},
                                "model_key": args.model_key,
                                "model_id": model_id,
                                "prompt_hash": stable_hash(meta["prompt"]),
                                "text": gen["text"],
                                "output_hash": stable_hash(gen["text"]),
                                "generated_tokens": gen["generated_tokens"],
                                "terminated_by_eos": gen["terminated_by_eos"],
                                "truncated_at_max_tokens": gen["truncated_at_max_tokens"],
                            }
                            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        f.flush()

    expected = len(datasets) * n * len(scopes) * len(policies)
    count = 0
    for path in (output_root / "outputs").glob("*/*/*.jsonl"):
        count += len(load_jsonl(path))
    if count != expected:
        raise AssertionError(f"{args.model_key}: expected {expected} outputs, got {count}")
    print(f"{args.model_key}: complete {count}/{expected}")


if __name__ == "__main__":
    main()
