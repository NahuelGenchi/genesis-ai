from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .ingest import sha256_file
from .model import GenesisLM

EXPERIMENT_VERSION = "m6-scale-5m-rope-v1"
FINALIST_VERSION = "m6-architecture-finalist-v1"
PREFLIGHT_VERSION = "m6-scale-5m-rope-preflight-v1"
LADDER_SUITES = (
    Path("evals/m6-domain-selection-v2.json"),
    Path("evals/m6-domain-ladder-d2-v1.json"),
    Path("evals/m6-domain-ladder-d3-v1.json"),
    Path("evals/m6-domain-ladder-d4-v1.json"),
    Path("evals/m6-domain-ladder-d5-v1.json"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_scale_contract(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ModelConfig]:
    experiment_path = Path(experiment_path)
    finalist_path = Path(finalist_path)
    preflight_path = Path(preflight_path)
    experiment = load_json(experiment_path)
    finalist = load_json(finalist_path)
    preflight = load_json(preflight_path)

    if experiment.get("format_version") != "1.0" or experiment.get("name") != EXPERIMENT_VERSION:
        raise ValueError("unsupported ~5M scale experiment")
    if experiment.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("~5M scale experiment violates zero-cash contract")
    evidence = experiment.get("architecture_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("architecture evidence contract is missing")

    if finalist.get("format_version") != "1.0" or finalist.get("finalist_version") != FINALIST_VERSION:
        raise ValueError("unsupported architecture finalist evidence")
    if finalist.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("architecture finalist evidence violates zero-cash contract")
    if evidence.get("finalist_version") != FINALIST_VERSION:
        raise ValueError("experiment/finalist version mismatch")
    if finalist.get("accepted_architecture") != evidence.get("accepted_architecture") or finalist.get("accepted_architecture") != "rope-only":
        raise ValueError("~5M experiment requires reproduced rope-only architecture")
    decision = finalist.get("decision")
    if not isinstance(decision, dict) or decision.get("passed") is not True:
        raise ValueError("architecture finalist did not pass fresh-seed reproduction")

    if preflight.get("format_version") != "1.0" or preflight.get("preflight_version") != PREFLIGHT_VERSION:
        raise ValueError("unsupported ~5M preflight evidence")
    if preflight.get("cash_compute_cost_usd") != 0.0 or preflight.get("promotion_authority") is not False:
        raise ValueError("~5M preflight evidence violates research-only zero-cash contract")
    if evidence.get("preflight_version") != PREFLIGHT_VERSION:
        raise ValueError("experiment/preflight version mismatch")
    projection = preflight.get("projection")
    if not isinstance(projection, dict) or projection.get("passes_remote_cpu_time_budget") is not True:
        raise ValueError("~5M full training is not authorized on GitHub-hosted CPU")
    if preflight.get("accepted_architecture") != "rope-only":
        raise ValueError("preflight architecture differs from reproduced finalist")

    model_block = experiment.get("model")
    if not isinstance(model_block, dict) or not isinstance(model_block.get("config"), dict):
        raise ValueError("~5M model definition is missing")
    config = ModelConfig.from_dict(model_block["config"])
    model = GenesisLM(config)
    expected_parameters = int(model_block.get("expected_parameter_count", -1))
    if expected_parameters != 4_954_624 or model.parameter_count() != expected_parameters:
        raise ValueError("~5M parameter-count contract drifted")
    if config.position_encoding != "rotary" or config.context_length != 128 or config.vocab_size != 512:
        raise ValueError("~5M architecture contract drifted")
    if preflight.get("parameter_count") != expected_parameters or preflight.get("config") != config.to_dict():
        raise ValueError("preflight did not measure the exact frozen ~5M model")

    training = experiment.get("training")
    if not isinstance(training, dict):
        raise ValueError("~5M training contract is missing")
    if training.get("initialization") != "genesis-random-from-scratch":
        raise ValueError("~5M model must be initialized from scratch")
    if int(training.get("target_training_tokens", 0)) != 20_000_000:
        raise ValueError("~5M training budget drifted")
    if int(training.get("examples_per_domain", 0)) != 8_192:
        raise ValueError("~5M verifier-record breadth drifted")
    if float(training.get("procedural_step_fraction", -1.0)) != 0.4 or float(training.get("public_step_fraction", -1.0)) != 0.6:
        raise ValueError("~5M training mix must remain exactly 40/60")
    if training.get("mandatory_first_and_terminator_coverage") is not True or training.get("unique_target_contexts_only") is not True:
        raise ValueError("~5M training must preserve explicit stop coverage and unique contexts")

    return experiment, finalist, preflight, config


def evidence_hashes(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
) -> dict[str, str]:
    return {
        "experiment_sha256": sha256_file(experiment_path),
        "architecture_finalist_sha256": sha256_file(finalist_path),
        "scale_preflight_sha256": sha256_file(preflight_path),
    }
