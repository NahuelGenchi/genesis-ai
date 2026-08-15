from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .filtering import exact_fingerprint, quality_reason
from .tokenizer import ByteBPETokenizer
from .verifiers import VERIFIER_VERSION, verify_task

FARM_VERSION = "cpu-screen-v1"
ALLOWED_RUNNER = "ubuntu-latest"
MODEL_LANES = {"architecture", "optimizer", "tiny-model"}
EXPENSIVE_STAGE_LANES = {"architecture", "optimizer", "tiny-model", "tokenizer", "data-filtering"}
REQUIRED_LANES = {"architecture", "optimizer", "tiny-model", "tokenizer", "data-filtering", "evaluation", "verifier"}


def load_definition(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "farm_version",
        "runner",
        "public_only",
        "paid_runners_allowed",
        "screening_only",
        "max_jobs",
        "max_parallel",
        "timeout_minutes",
        "seed",
        "lanes",
        "candidates",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("CPU farm definition has unexpected fields")
    if raw["format_version"] != "1.0" or raw["farm_version"] != FARM_VERSION:
        raise ValueError("unsupported CPU farm definition")
    if raw["runner"] != ALLOWED_RUNNER:
        raise ValueError("CPU farm must use ubuntu-latest")
    if raw["public_only"] is not True or raw["paid_runners_allowed"] is not False:
        raise ValueError("CPU farm must be public-only and forbid paid runners")
    if raw["screening_only"] is not True:
        raise ValueError("CPU farm results must be screening-only")
    max_jobs = int(raw["max_jobs"])
    max_parallel = int(raw["max_parallel"])
    timeout = int(raw["timeout_minutes"])
    if max_jobs <= 0 or max_jobs > 20:
        raise ValueError("max_jobs must be in [1, 20]")
    if max_parallel <= 0 or max_parallel > 12 or max_parallel > max_jobs:
        raise ValueError("max_parallel must be in [1, 12] and <= max_jobs")
    if timeout <= 0 or timeout > 12:
        raise ValueError("timeout_minutes must be in [1, 12]")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ValueError("seed must be an integer")
    lanes = raw["lanes"]
    if not isinstance(lanes, dict) or set(lanes) != REQUIRED_LANES:
        raise ValueError("CPU farm lanes do not match the required lane set")
    for name, lane in lanes.items():
        if not isinstance(lane, dict) or set(lane) != {"kind", "objective", "threshold_fraction"}:
            raise ValueError(f"invalid lane config: {name}")
        if lane["kind"] not in {"candidate", "guard"}:
            raise ValueError(f"invalid lane kind: {name}")
        if lane["objective"] not in {"minimize", "maximize"}:
            raise ValueError(f"invalid lane objective: {name}")
        if float(lane["threshold_fraction"]) < 0:
            raise ValueError(f"negative threshold: {name}")
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or not candidates or len(candidates) > max_jobs:
        raise ValueError("invalid candidate count")
    seen: set[tuple[str, str]] = set()
    baselines: dict[str, int] = {name: 0 for name in REQUIRED_LANES}
    counts: dict[str, int] = {name: 0 for name in REQUIRED_LANES}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"lane", "variant", "baseline"}:
            raise ValueError("invalid CPU farm candidate")
        lane = candidate["lane"]
        variant = candidate["variant"]
        baseline = candidate["baseline"]
        if lane not in REQUIRED_LANES or not isinstance(variant, str) or not variant or not isinstance(baseline, bool):
            raise ValueError("invalid CPU farm candidate values")
        key = (lane, variant)
        if key in seen:
            raise ValueError(f"duplicate CPU farm candidate: {lane}/{variant}")
        seen.add(key)
        counts[lane] += 1
        baselines[lane] += int(baseline)
    for lane, config in lanes.items():
        if counts[lane] < 1:
            raise ValueError(f"lane has no candidates: {lane}")
        expected_baselines = 1 if config["kind"] == "candidate" else 0
        if baselines[lane] != expected_baselines:
            raise ValueError(f"lane baseline count invalid: {lane}")
    return raw


def matrix(definition: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seed = int(definition["seed"])
    include = []
    for item in definition["candidates"]:
        include.append(
            {
                "lane": item["lane"],
                "variant": item["variant"],
                "seed": seed,
                "needs_torch": item["lane"] in MODEL_LANES,
            }
        )
    return {"include": include}


def _model_config(lane: str, variant: str):
    from .config import ModelConfig

    common = dict(vocab_size=256, context_length=32, d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.0)
    if lane == "architecture":
        if variant == "layernorm-gelu":
            return ModelConfig(**common, norm_type="layernorm", dense_activation="gelu")
        if variant == "rmsnorm-gelu":
            return ModelConfig(**common, norm_type="rmsnorm", dense_activation="gelu")
        if variant == "layernorm-swiglu":
            return ModelConfig(**{**common, "d_ff": 341}, norm_type="layernorm", dense_activation="swiglu")
    if lane == "optimizer":
        if variant in {"adamw", "sgd"}:
            return ModelConfig(**common)
    if lane == "tiny-model":
        variants = {
            "tiny-96x2": dict(vocab_size=256, context_length=32, d_model=96, n_heads=4, n_layers=2, d_ff=384, dropout=0.0),
            "tiny-128x2": common,
            "tiny-128x3": dict(vocab_size=256, context_length=32, d_model=128, n_heads=4, n_layers=3, d_ff=512, dropout=0.0),
        }
        if variant in variants:
            return ModelConfig(**variants[variant])
    raise ValueError(f"unknown model variant: {lane}/{variant}")


def _synthetic_stream(length: int, *, offset: int = 0) -> list[int]:
    values: list[int] = []
    for i in range(offset, offset + length):
        block = (i // 29) % 11
        position = i % 29
        value = (position * 7 + block * 13 + (position % 5) * block) % 128
        values.append(value)
    return values


def _fixed_validation_loss(model, stream: list[int], context: int, torch) -> float:
    starts = list(range(0, min(1024, len(stream) - context - 1), max(1, context // 2)))[:24]
    xs = [stream[start : start + context] for start in starts]
    ys = [stream[start + 1 : start + context + 1] for start in starts]
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    assert loss is not None
    return float(loss.detach().cpu())


def _run_model_screen(lane: str, variant: str, seed: int) -> dict[str, Any]:
    import torch

    from .model import GenesisLM
    from .research import estimated_training_flops_per_token

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    config = _model_config(lane, variant)
    config.validate()
    model = GenesisLM(config)
    flops_per_token = estimated_training_flops_per_token(model)
    tokens_per_step = 256
    training_flop_budget = 12_000_000_000
    steps = max(1, training_flop_budget // (flops_per_token * tokens_per_step))
    batch_size = max(1, tokens_per_step // config.context_length)
    actual_tokens_per_step = batch_size * config.context_length

    if lane == "optimizer" and variant == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        learning_rate = 0.05
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False, fused=False)
        learning_rate = 0.003

    train_stream = _synthetic_stream(12_000)
    validation_stream = _synthetic_stream(3_000, offset=20_000)
    initial = _fixed_validation_loss(model, validation_stream, config.context_length, torch)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    started = time.perf_counter()
    model.train()
    last_loss = float("nan")
    max_start = len(train_stream) - config.context_length - 1
    for _ in range(int(steps)):
        starts = torch.randint(0, max_start, (batch_size,), generator=generator).tolist()
        x = torch.tensor([train_stream[s : s + config.context_length] for s in starts], dtype=torch.long)
        y = torch.tensor([train_stream[s + 1 : s + config.context_length + 1] for s in starts], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - started
    final = _fixed_validation_loss(model, validation_stream, config.context_length, torch)
    return {
        "metric_name": "synthetic_validation_loss",
        "metric_value": final,
        "objective": "minimize",
        "initial_metric_value": initial,
        "last_training_loss": last_loss,
        "parameter_count": model.parameter_count(),
        "active_parameter_count": model.estimated_active_parameter_count(),
        "training_flop_budget": training_flop_budget,
        "estimated_training_flops": int(steps) * actual_tokens_per_step * flops_per_token,
        "training_flops_per_token": flops_per_token,
        "steps": int(steps),
        "tokens_seen": int(steps) * actual_tokens_per_step,
        "learning_rate": learning_rate,
        "training_seconds": elapsed,
        "training_tokens_per_second": (int(steps) * actual_tokens_per_step) / elapsed,
        "config": config.to_dict(),
    }


def _tokenizer_texts() -> list[str]:
    return [
        "Genesis learns from deterministic experiments and keeps exact provenance for every result.\n",
        "La evaluación debe separar los datos de entrenamiento de los conjuntos de prueba privados.\n",
        "Les expériences reproductibles permettent de comparer la qualité à calcul constant.\n",
        "def transform(x: int) -> int:\n    return 3 * x * x - 7 * x + 11\n",
        "For x=-13, y=29, compute 4*x - 3*y + 17 and return only the integer.\n",
    ] * 8


def _run_tokenizer_screen(variant: str) -> dict[str, Any]:
    current = ByteBPETokenizer.load(Path("tokenizers/genesis-v0.json"))
    if variant == "genesis-v0":
        tokenizer = current
    elif variant == "genesis-v0-trim384":
        tokenizer = ByteBPETokenizer(tuple(current.merges[:128]))
    elif variant == "byte-256":
        tokenizer = ByteBPETokenizer(())
    else:
        raise ValueError(f"unknown tokenizer variant: {variant}")
    texts = _tokenizer_texts()
    utf8_bytes = sum(len(text.encode("utf-8")) for text in texts)
    tokens = 0
    for text in texts:
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise AssertionError("tokenizer round-trip failure")
        tokens += len(ids)
    bytes_per_token = utf8_bytes / tokens
    return {
        "metric_name": "bytes_per_token",
        "metric_value": bytes_per_token,
        "objective": "maximize",
        "vocab_size": tokenizer.vocab_size,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "round_trip_failures": 0,
    }


def _filter_fixture() -> list[tuple[str, bool]]:
    good = [
        "A carefully written paragraph with enough semantic content to be useful for language-model training and evaluation.",
        "Deterministic experiments need stable inputs, explicit metrics, and reproducible decisions across independent runs.",
        "La calidad del corpus depende de procedencia clara, deduplicación exacta y filtros conservadores de contenido.",
        "Les données propres et traçables permettent de mesurer les progrès sans confondre bruit et apprentissage réel.",
        "A compact code explanation can still contain enough useful structure to teach syntax, names, control flow, and intent.",
    ]
    bad = [
        "short noisy item that should not survive",
        "x" * 140,
        "valid words\x01but an embedded control character makes this document unsafe for the training corpus.",
        "repeat me\nrepeat me\nrepeat me\nrepeat me\nrepeat me\n",
        good[0],
    ]
    return [(text, True) for text in good] + [(text, False) for text in bad]


def _run_filter_screen(variant: str) -> dict[str, Any]:
    mins = {"min-chars-20": 20, "min-chars-40": 40, "min-chars-80": 80}
    if variant not in mins:
        raise ValueError(f"unknown filtering variant: {variant}")
    min_chars = mins[variant]
    seen: set[str] = set()
    tp = fp = fn = tn = 0
    for text, expected_keep in _filter_fixture():
        reason = quality_reason(text, min_chars=min_chars)
        if reason is None:
            fingerprint = exact_fingerprint(text)
            if fingerprint in seen:
                reason = "exact_duplicate"
            else:
                seen.add(fingerprint)
        predicted_keep = reason is None
        if predicted_keep and expected_keep:
            tp += 1
        elif predicted_keep and not expected_keep:
            fp += 1
        elif not predicted_keep and expected_keep:
            fn += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "metric_name": "fixture_f1",
        "metric_value": f1,
        "objective": "maximize",
        "min_chars": min_chars,
        "precision": precision,
        "recall": recall,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _run_evaluation_guard(variant: str) -> dict[str, Any]:
    if variant == "m3-contract":
        suite = json.loads(Path("evals/m3-v1.json").read_text(encoding="utf-8"))
        checks = [
            suite.get("suite_version") == "m3-v1",
            suite.get("comparison", {}).get("primary_metric") == "validation_loss",
            suite.get("comparison", {}).get("lower_is_better") is True,
            suite.get("contamination", {}).get("block_exact_overlap") is True,
            int(suite.get("validation", {}).get("batches", 0)) > 0,
        ]
    elif variant == "domain-v2-contract":
        suite = json.loads(Path("evals/m6-domain-selection-v2.json").read_text(encoding="utf-8"))
        checks = [
            suite.get("suite_version") == "m6-domain-selection-v2",
            suite.get("domains") == ["math", "structured", "code"],
            suite.get("termination", {}).get("required") is True,
            suite.get("termination", {}).get("delimiter") == "\n",
            int(suite.get("tasks_per_domain", 0)) >= 60,
        ]
    else:
        raise ValueError(f"unknown evaluation guard: {variant}")
    score = sum(bool(value) for value in checks) / len(checks)
    return {"metric_name": "contract_pass_fraction", "metric_value": score, "objective": "maximize", "checks": len(checks)}


def _run_verifier_guard(variant: str) -> dict[str, Any]:
    version = VERIFIER_VERSION
    if variant == "integer-exact":
        task = {"verifier": {"version": version, "kind": "integer_exact", "expected": 42}}
        cases = [("42", True), ("41", False), ("42.0", False), ("answer: 42", False)]
    elif variant == "json-exact":
        task = {"verifier": {"version": version, "kind": "json_exact", "expected": {"a": [1, 2], "ok": True}}}
        cases = [('{"a":[1,2],"ok":true}', True), ('{"ok":true,"a":[1,2]}', True), ('{"a":[2,1],"ok":true}', False), ('not json', False)]
    elif variant == "restricted-expression":
        task = {
            "verifier": {
                "version": version,
                "kind": "restricted_expression",
                "tests": [
                    {"variables": {"x": 2}, "expected": 7},
                    {"variables": {"x": -3}, "expected": -8},
                ],
            }
        }
        cases = [("3*x+1", True), ("x+1", False), ('__import__("os").system("echo unsafe")', False), ("x/2", False)]
    else:
        raise ValueError(f"unknown verifier guard: {variant}")
    correct = 0
    for answer, expected in cases:
        observed = verify_task(task, answer).passed
        correct += int(observed is expected)
    score = correct / len(cases)
    return {"metric_name": "classification_accuracy", "metric_value": score, "objective": "maximize", "cases": len(cases)}


def run_screen(definition: dict[str, Any], *, lane: str, variant: str, seed: int) -> dict[str, Any]:
    declared = {(item["lane"], item["variant"]) for item in definition["candidates"]}
    if (lane, variant) not in declared:
        raise ValueError(f"undeclared CPU screen: {lane}/{variant}")
    if seed != int(definition["seed"]):
        raise ValueError("screen seed does not match frozen definition")
    if lane in MODEL_LANES:
        metrics = _run_model_screen(lane, variant, seed)
    elif lane == "tokenizer":
        metrics = _run_tokenizer_screen(variant)
    elif lane == "data-filtering":
        metrics = _run_filter_screen(variant)
    elif lane == "evaluation":
        metrics = _run_evaluation_guard(variant)
    elif lane == "verifier":
        metrics = _run_verifier_guard(variant)
    else:
        raise ValueError(f"unsupported CPU farm lane: {lane}")
    return {
        "format_version": "1.0",
        "farm_version": FARM_VERSION,
        "lane": lane,
        "variant": variant,
        "seed": seed,
        "screening_only": True,
        "promotion_eligible": False,
        "cash_compute_cost_usd": 0.0,
        "runner_contract": ALLOWED_RUNNER,
        "metrics": metrics,
    }


def _relative_improvement(objective: str, baseline: float, candidate: float) -> float:
    denominator = max(abs(baseline), 1e-12)
    if objective == "minimize":
        return (baseline - candidate) / denominator
    return (candidate - baseline) / denominator


def aggregate(definition: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(item["lane"], item["variant"]) for item in definition["candidates"]}
    observed = {(item.get("lane"), item.get("variant")) for item in results}
    if observed != expected or len(results) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"CPU farm result set mismatch; missing={missing} extra={extra}")
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("farm_version") != FARM_VERSION or result.get("screening_only") is not True or result.get("promotion_eligible") is not False:
            raise ValueError("invalid CPU farm result contract")
        by_lane.setdefault(str(result["lane"]), []).append(result)

    lane_summaries: dict[str, Any] = {}
    expensive_stage_eligible: list[dict[str, Any]] = []
    for lane, lane_config in definition["lanes"].items():
        items = by_lane[lane]
        objective = lane_config["objective"]
        if lane_config["kind"] == "guard":
            passed = all(float(item["metrics"]["metric_value"]) == 1.0 for item in items)
            lane_summaries[lane] = {"kind": "guard", "passed": passed, "results": sorted((item["variant"], item["metrics"]["metric_value"]) for item in items)}
            continue
        candidate_defs = [item for item in definition["candidates"] if item["lane"] == lane]
        baseline_name = next(item["variant"] for item in candidate_defs if item["baseline"])
        baseline_result = next(item for item in items if item["variant"] == baseline_name)
        key = lambda item: float(item["metrics"]["metric_value"])
        winner = min(items, key=key) if objective == "minimize" else max(items, key=key)
        baseline_value = float(baseline_result["metrics"]["metric_value"])
        winner_value = float(winner["metrics"]["metric_value"])
        improvement = _relative_improvement(objective, baseline_value, winner_value)
        threshold = float(lane_config["threshold_fraction"])
        survives = winner["variant"] != baseline_name and improvement >= threshold
        lane_summaries[lane] = {
            "kind": "candidate",
            "objective": objective,
            "baseline": baseline_name,
            "baseline_metric": baseline_value,
            "winner": winner["variant"],
            "winner_metric": winner_value,
            "improvement_fraction": improvement,
            "threshold_fraction": threshold,
            "survives_cpu_screen": survives,
        }
        if survives and lane in EXPENSIVE_STAGE_LANES:
            expensive_stage_eligible.append(
                {
                    "lane": lane,
                    "variant": winner["variant"],
                    "improvement_fraction": improvement,
                    "source": "cpu-screen-v1",
                }
            )

    guards_pass = all(summary.get("passed", True) for summary in lane_summaries.values())
    return {
        "format_version": "1.0",
        "farm_version": FARM_VERSION,
        "screening_only": True,
        "promotion_eligible": False,
        "cash_compute_cost_usd": 0.0,
        "runner_contract": ALLOWED_RUNNER,
        "job_count": len(results),
        "guards_pass": guards_pass,
        "lane_summaries": lane_summaries,
        "expensive_stage_eligible": expensive_stage_eligible if guards_pass else [],
        "gpu_policy": {
            "eligible_only_after_cpu_screen": True,
            "screening_result_can_promote_checkpoint": False,
            "full_reproduction_and_frozen_evaluation_required_before_promotion": True,
        },
    }


def _read_result_dir(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.json")):
        results.append(json.loads(file.read_text(encoding="utf-8")))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis public GitHub CPU research farm")
    sub = parser.add_subparsers(dest="command", required=True)

    p_matrix = sub.add_parser("matrix")
    p_matrix.add_argument("--definition", type=Path, required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--definition", type=Path, required=True)
    p_run.add_argument("--lane", required=True)
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--output", type=Path, required=True)

    p_aggregate = sub.add_parser("aggregate")
    p_aggregate.add_argument("--definition", type=Path, required=True)
    p_aggregate.add_argument("--input-dir", type=Path, required=True)
    p_aggregate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    definition = load_definition(args.definition)
    if args.command == "matrix":
        print(json.dumps(matrix(definition), separators=(",", ":"), sort_keys=True))
        return
    if args.command == "run":
        result = run_screen(definition, lane=args.lane, variant=args.variant, seed=args.seed)
    else:
        result = aggregate(definition, _read_result_dir(args.input_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
