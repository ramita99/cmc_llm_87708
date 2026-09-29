from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cmc87708.evaluation import protected_detector_audit
from cmc87708.prompts import basic_prompt, deterministic_template, proposed_prompt
from cmc87708.protocol import assert_llm_facing_packet, stable_hash

ALL_METHODS = ["deterministic_template", "basic_prompt", "proposed_source_gated"]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 generation requires a CUDA GPU for this experiment.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected GPU does not support BF16 efficiently. "
                "Use an L4/A100-class runtime for the CMC 87708 generation protocol."
            )
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_model(model_id: str, dtype_name: str, attention_impl: str):
    dtype = resolve_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if attention_impl:
        kwargs["attn_implementation"] = attention_impl
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def _normalize_eos_ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _decode_generation(
    tokenizer,
    generated_tokens: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> dict:
    ids = generated_tokens.tolist()
    eos_ids = set(eos_token_ids)
    pad_id = tokenizer.pad_token_id

    terminal_pos = None
    terminal_token = None
    for idx, token_id in enumerate(ids):
        if token_id in eos_ids or (pad_id is not None and token_id == pad_id):
            terminal_pos = idx
            terminal_token = int(token_id)
            break

    if terminal_pos is None:
        effective_ids = ids
        terminated = False
    else:
        effective_ids = ids[:terminal_pos]
        terminated = True

    generated_count = len(effective_ids)
    truncated = (not terminated) and len(ids) >= max_new_tokens
    text = tokenizer.decode(effective_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_tokens": generated_count,
        "terminated_by_eos": terminated,
        "termination_token_id": terminal_token,
        "truncated_at_max_tokens": truncated,
        "eos_token_ids_used": eos_token_ids,
    }


def _generate_batch_once(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    rendered = [render_chat(tokenizer, p) for p in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    # Cross-model sensitivity protocol: honor each model's published
    # generation EOS configuration instead of overriding it with a tokenizer-
    # only EOS id. This supports scalar or multiple valid EOS ids.
    eos_value = model.generation_config.eos_token_id
    eos_token_ids = _normalize_eos_ids(eos_value)
    if not eos_token_ids:
        eos_token_ids = _normalize_eos_ids(tokenizer.eos_token_id)
    eos_for_generate = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_for_generate,
            use_cache=True,
        )
    prefix_len = inputs["input_ids"].shape[-1]
    return [
        _decode_generation(tokenizer, seq[prefix_len:], max_new_tokens, eos_token_ids)
        for seq in output
    ]


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """Generate a batch; automatically split a batch if CUDA memory is exhausted."""
    try:
        return _generate_batch_once(model, tokenizer, prompts, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        if len(prompts) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(prompts) // 2
        return (
            generate_batch(model, tokenizer, prompts[:mid], max_new_tokens)
            + generate_batch(model, tokenizer, prompts[mid:], max_new_tokens)
        )


def batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def packet_key(packet: dict) -> str:
    t = packet.get("traceability", {})
    return "|".join(map(str, [
        packet.get("dataset"), t.get("window_id"), t.get("edge_id"), packet.get("state")
    ]))


def prompt_for(method: str, packet: dict) -> str | None:
    assert_llm_facing_packet(packet, f"{method} prompt packet")
    if method == "proposed_source_gated":
        return proposed_prompt(packet)
    if method == "basic_prompt":
        return basic_prompt(packet)
    if method == "deterministic_template":
        return None
    raise ValueError(method)


def filtered_records(method: str, records: list[dict], exp: dict) -> list[dict]:
    if method in {"basic_prompt", "deterministic_template"}:
        return [r for r in records if r["state"] == exp["full_state_for_method_comparison"]]
    return records


def _base_result(packet: dict, method: str, model_id: str | None, prompt: str | None) -> dict:
    return {
        "record_key": packet_key(packet),
        "dataset": packet["dataset"],
        "state": packet["state"],
        "method": method,
        "model": model_id,
        # Hashes are audit metadata only and are deliberately outside `packet`.
        "packet_hash": stable_hash(packet),
        "prompt_hash": None if prompt is None else stable_hash(prompt),
        "packet": packet,
    }


def write_method_outputs(
    method: str,
    packet_files: list[Path],
    exp: dict,
    root: Path,
    model_id: str,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    resume: bool,
) -> None:
    out_dir = root / "outputs" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_tolerance = float(exp.get("numeric_tolerance", 1e-4))
    fallback_on_truncation = bool(exp.get("fallback_on_truncation", True))

    for packet_file in packet_files:
        records = filtered_records(method, load_jsonl(packet_file), exp)
        out_file = out_dir / packet_file.name

        completed: set[str] = set()
        if resume and out_file.exists():
            completed = {
                r.get("record_key") or packet_key(r["packet"])
                for r in load_jsonl(out_file)
            }
        pending = [p for p in records if packet_key(p) not in completed]
        mode = "a" if resume and out_file.exists() else "w"

        with out_file.open(mode, encoding="utf-8") as f:
            if method == "deterministic_template":
                iterator = tqdm(pending, desc=f"{method}:{packet_file.stem}")
                for packet in iterator:
                    assert_llm_facing_packet(packet, "template packet")
                    text = deterministic_template(packet)
                    result = _base_result(packet, method, None, None)
                    result.update({
                        "native_text": text,
                        "text": text,
                        "native_output_hash": stable_hash(text),
                        "output_hash": stable_hash(text),
                        "generated_tokens": None,
                        "terminated_by_eos": None,
                        "truncated_at_max_tokens": False,
                        "audit_pass": True,
                        "audit_failures": [],
                        "fallback_used": False,
                        "fallback_reason": None,
                    })
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                continue

            total_batches = (len(pending) + batch_size - 1) // batch_size
            iterator = tqdm(batches(pending, batch_size), total=total_batches,
                            desc=f"{method}:{packet_file.stem}")
            for packet_batch in iterator:
                for packet in packet_batch:
                    assert_llm_facing_packet(packet, "generation packet")
                prompts = [prompt_for(method, p) for p in packet_batch]
                generations = generate_batch(model, tokenizer, prompts, max_new_tokens)

                for packet, prompt, gen in zip(packet_batch, prompts, generations):
                    native_text = gen["text"]
                    audit = {"pass": True, "failures": [], "metrics": {}}
                    fallback_used = False
                    fallback_reason = None
                    final_text = native_text

                    if method == "proposed_source_gated":
                        audit = protected_detector_audit(
                            packet, native_text, numeric_tolerance=numeric_tolerance
                        )
                        reasons = list(audit["failures"])
                        if gen["truncated_at_max_tokens"] and fallback_on_truncation:
                            reasons.append("max_token_truncation")
                        if reasons:
                            fallback_used = True
                            fallback_reason = ";".join(reasons)
                            final_text = deterministic_template(packet)

                    result = _base_result(packet, method, model_id, prompt)
                    result.update({
                        "native_text": native_text,
                        "text": final_text,
                        "native_output_hash": stable_hash(native_text),
                        "output_hash": stable_hash(final_text),
                        "generated_tokens": gen["generated_tokens"],
                        "terminated_by_eos": gen["terminated_by_eos"],
                        "termination_token_id": gen.get("termination_token_id"),
                        "eos_token_ids_used": gen.get("eos_token_ids_used"),
                        "truncated_at_max_tokens": gen["truncated_at_max_tokens"],
                        "audit_pass": bool(audit["pass"] and not gen["truncated_at_max_tokens"]),
                        "audit_failures": audit["failures"],
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                    })
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/primary_final.yaml")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--method", choices=ALL_METHODS, default=None,
                    help="Backward-compatible single-method mode.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Methods to run, or 'all'. Model is loaded only once.")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--attention-impl", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    exp = load_yaml(args.experiment)
    root = Path(exp["output_root"])
    packet_files = sorted((root / "packets").glob("*.jsonl"))
    if not packet_files:
        raise FileNotFoundError(f"No packet JSONL files found under {root / 'packets'}")

    if args.method:
        methods = [args.method]
    elif args.methods:
        methods = ALL_METHODS if args.methods == ["all"] else args.methods
    else:
        methods = ALL_METHODS
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")

    batch_size = int(args.batch_size or exp.get("generation_batch_size", 4))
    max_new_tokens = int(args.max_new_tokens or exp.get("max_new_tokens", 384))
    dtype_name = str(args.dtype or exp.get("generation_dtype", "bfloat16"))
    attention_impl = str(args.attention_impl or exp.get("attention_implementation", "sdpa"))

    if "deterministic_template" in methods:
        write_method_outputs(
            "deterministic_template", packet_files, exp, root, args.model,
            None, None, batch_size, max_new_tokens, args.resume,
        )

    llm_methods = [m for m in methods if m != "deterministic_template"]
    if llm_methods:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for Gemma generation.")
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GiB | "
            f"dtype={dtype_name} | batch_size={batch_size} | max_new_tokens={max_new_tokens}"
        )
        model, tokenizer = load_model(args.model, dtype_name, attention_impl)
        print(
            "Generation EOS config:",
            model.generation_config.eos_token_id,
            "| tokenizer eos:",
            tokenizer.eos_token_id,
            "| tokenizer pad:",
            tokenizer.pad_token_id,
        )
        for method in llm_methods:
            write_method_outputs(
                method, packet_files, exp, root, args.model,
                model, tokenizer, batch_size, max_new_tokens, args.resume,
            )


if __name__ == "__main__":
    main()
