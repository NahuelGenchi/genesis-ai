from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FUNNEL_VERSION = "weak-domain-successive-halving-v1"
NORMAL_TOKEN_BUDGET = 3_000_000
TINY_TOKEN_BUDGET = 225_000
MEDIUM_TOKEN_BUDGET = 750_000
FULL_TOKEN_BUDGET = NORMAL_TOKEN_BUDGET
TINY_SURVIVORS = 3
MEDIUM_SURVIVORS = 1

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "structured-full-sort",
        "domains": ["structured"],
        "supervision": "full transformation target",
    },
    {
        "id": "structured-pairwise-rank",
        "domains": ["structured"],
        "supervision": "pairwise comparison and rank decomposition",
    },
    {
        "id": "structured-prefix-next",
        "domains": ["structured"],
        "supervision": "prefix construction and next-element prediction",
    },
    {
        "id": "structured-partial-completion",
        "domains": ["structured"],
        "supervision": "partial sorted-sequence completion",
    },
    {
        "id": "structured-length-progression",
        "domains": ["structured"],
        "supervision": "response-length progression from short to full transformations",
    },
    {
        "id": "structured-mixed-decomposition",
        "domains": ["structured"],
        "supervision": "mixed decomposition plus full transformation",
    },
    {
        "id": "math-operation-level",
        "domains": ["math"],
        "supervision": "operation-level and intermediate deterministic supervision",
    },
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _seed_for(variant_id: str) -> int:
    digest = hashlib.sha256(f"{FUNNEL_VERSION}:{variant_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_000_000_000


def build_catalog() -> dict[str, Any]:
    variants = []
    for item in VARIANTS:
        variant = dict(item)
        variant.update(
            {
                "tiny_token_budget": TINY_TOKEN_BUDGET,
                "medium_token_budget": MEDIUM_TOKEN_BUDGET,
                "full_token_budget": FULL_TOKEN_BUDGET,
                "training_seed": _seed_for(str(item["id"])),
                "screening_only": True,
                "promotion_authority": False,
            }
        )
        variants.append(variant)

    catalog: dict[str, Any] = {
        "format_version": "1.0",
        "funnel_version": FUNNEL_VERSION,
        "normal_token_budget": NORMAL_TOKEN_BUDGET,
        "stages": {
            "tiny": {
                "token_budget": TINY_TOKEN_BUDGET,
                "fraction_of_normal": TINY_TOKEN_BUDGET / NORMAL_TOKEN_BUDGET,
                "candidate_count": len(variants),
                "survivors": TINY_SURVIVORS,
            },
            "medium": {
                "token_budget": MEDIUM_TOKEN_BUDGET,
                "fraction_of_normal": MEDIUM_TOKEN_BUDGET / NORMAL_TOKEN_BUDGET,
                "candidate_count": TINY_SURVIVORS,
                "survivors": MEDIUM_SURVIVORS,
            },
            "full": {
                "token_budget": FULL_TOKEN_BUDGET,
                "fraction_of_normal": 1.0,
                "candidate_count": MEDIUM_SURVIVORS,
                "survivors": MEDIUM_SURVIVORS,
            },
        },
        "selection_contract": {
            "primary_metric": "weak_domain_gain_pp",
            "secondary_metric": "code_retention_pp",
            "tertiary_metric": "development_oracle_loss",
            "equal_processed_tokens_required": True,
            "holdout_prompt_overlap_must_be_zero": True,
            "screening_metrics_have_promotion_authority": False,
        },
        "artifact_contract": {
            "checkpoint_before_downstream_evaluation": True,
            "hash_addressed_checkpoint_required": True,
            "resume_from_exact_artifact_after_downstream_failure": True,
        },
        "safety_contract": {
            "cash_compute_cost_usd": 0.0,
            "paid_api_forbidden": True,
            "external_proprietary_weights_forbidden": True,
            "user_hardware_forbidden": True,
            "frozen_promotion_gate_changes_forbidden": True,
            "screening_only": True,
            "promotion_authority": False,
        },
        "variants": variants,
    }
    catalog["catalog_sha256"] = _sha256_object(catalog)
    return catalog


def validate_catalog(catalog: dict[str, Any]) -> None:
    stored = catalog.get("catalog_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError("catalog_sha256 is required")
    unhashed = dict(catalog)
    unhashed.pop("catalog_sha256", None)
    if _sha256_object(unhashed) != stored:
        raise ValueError("catalog hash mismatch")
    if catalog.get("funnel_version") != FUNNEL_VERSION:
        raise ValueError("unsupported funnel version")
    if catalog.get("safety_contract", {}).get("cash_compute_cost_usd") != 0.0:
        raise ValueError("funnel violates zero-cash contract")
    if catalog.get("safety_contract", {}).get("promotion_authority") is not False:
        raise ValueError("screening funnel cannot have promotion authority")
    variants = catalog.get("variants")
    if not isinstance(variants, list) or len(variants) != len(VARIANTS):
        raise ValueError("variant catalog must contain every predeclared family")
    ids = [str(item.get("id")) for item in variants]
    if len(set(ids)) != len(ids):
        raise ValueError("variant ids must be unique")
    tiny_fraction = float(catalog["stages"]["tiny"]["fraction_of_normal"])
    medium_fraction = float(catalog["stages"]["medium"]["fraction_of_normal"])
    if not 0.05 <= tiny_fraction <= 0.10:
        raise ValueError("tiny stage must remain within the predeclared 5-10% budget")
    if medium_fraction != 0.25:
        raise ValueError("medium stage must remain at the predeclared 25% budget")


def select_survivors(results: list[dict[str, Any]], *, keep: int) -> list[dict[str, Any]]:
    if keep <= 0 or keep > len(results):
        raise ValueError("invalid survivor count")
    processed = {int(item["processed_tokens"]) for item in results}
    if len(processed) != 1:
        raise ValueError("successive-halving comparison requires equal processed tokens")
    ids = [str(item["variant_id"]) for item in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate variant result")
    for item in results:
        if item.get("screening_only") is not True or item.get("promotion_authority") is not False:
            raise ValueError("screening result contract violated")
        if int(item.get("holdout_prompt_overlap_count", -1)) != 0:
            raise ValueError("blocking holdout overlap in screening result")

    ranked = sorted(
        results,
        key=lambda item: (
            -float(item["weak_domain_gain_pp"]),
            -float(item["code_retention_pp"]),
            float(item["development_oracle_loss"]),
            str(item["variant_id"]),
        ),
    )
    return ranked[:keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the weak-domain successive-halving research funnel.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog")
    catalog_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--catalog", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "catalog":
        catalog = build_catalog()
        validate_catalog(catalog)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "validate":
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict):
            raise ValueError("catalog must be a JSON object")
        validate_catalog(catalog)


if __name__ == "__main__":
    main()
