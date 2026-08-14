from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .verifiers import VERIFIER_VERSION

GENERATOR_VERSION = "procedural-v1"
DOMAINS = ("math", "structured", "code")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _task_id(task_without_id: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical(task_without_id).encode("utf-8")).hexdigest()
    return f"task-{digest[:20]}"


def _math_task(rng: random.Random, difficulty: int) -> tuple[str, dict[str, Any]]:
    limit = 10 ** min(difficulty + 1, 6)
    a = rng.randint(-limit, limit)
    b = rng.randint(-limit, limit)
    c = rng.randint(1, max(2, limit // 2))
    if difficulty <= 1:
        prompt = f"Return only the integer result of {a} + {b}."
        expected = a + b
    elif difficulty == 2:
        prompt = f"Return only the integer result of ({a}) * ({b}) + {c}."
        expected = a * b + c
    elif difficulty == 3:
        divisor = rng.randint(2, 12)
        product = a * divisor
        prompt = f"Return only the integer result of ({product}) // {divisor} + ({b}) * {c}."
        expected = product // divisor + b * c
    else:
        modulus = rng.randint(2, 97)
        prompt = f"Return only the integer result of (({a}) * ({b}) + {c}) % {modulus}."
        expected = (a * b + c) % modulus
    return prompt, {"kind": "integer_exact", "version": VERIFIER_VERSION, "expected": expected}


def _structured_task(rng: random.Random, difficulty: int) -> tuple[str, dict[str, Any]]:
    length = 4 + difficulty * 2
    values = [rng.randint(-50 * difficulty, 50 * difficulty) for _ in range(length)]
    mode = rng.choice(("sort", "unique_sort", "reverse")) if difficulty >= 3 else "sort"
    if mode == "sort":
        prompt = f"Return only JSON: sort this integer array ascending: {_canonical(values)}"
        expected: object = sorted(values)
    elif mode == "unique_sort":
        prompt = f"Return only JSON: remove duplicates, then sort ascending: {_canonical(values)}"
        expected = sorted(set(values))
    else:
        prompt = f"Return only JSON: reverse this array without changing its values: {_canonical(values)}"
        expected = list(reversed(values))
    return prompt, {"kind": "json_exact", "version": VERIFIER_VERSION, "expected": expected}


def _code_task(rng: random.Random, difficulty: int) -> tuple[str, dict[str, Any]]:
    a = rng.choice([value for value in range(-8, 9) if value != 0])
    b = rng.choice([value for value in range(-8, 9) if value != 0])
    c = rng.randint(-20, 20)
    if difficulty <= 2:
        formula = lambda x, y: a * x + b * y + c
        description = f"{a}*x + {b}*y + {c}"
    elif difficulty <= 4:
        d = rng.randint(2, 7)
        formula = lambda x, y: a * x * x + b * y + c * d
        description = f"{a}*x^2 + {b}*y + {c*d}"
    else:
        modulus = rng.randint(3, 23)
        formula = lambda x, y: (a * x * x + b * y + c) % modulus
        description = f"({a}*x^2 + {b}*y + {c}) modulo {modulus}"

    tests = []
    seen: set[tuple[int, int]] = set()
    while len(tests) < 8:
        pair = (rng.randint(-12, 12), rng.randint(-12, 12))
        if pair in seen:
            continue
        seen.add(pair)
        x, y = pair
        tests.append({"variables": {"x": x, "y": y}, "expected": formula(x, y)})
    prompt = (
        "Write only a restricted integer expression using variables x and y. "
        "Allowed operators: +, -, *, //, %, ** and parentheses. "
        f"For every integer x,y it must compute: {description}."
    )
    return prompt, {"kind": "restricted_expression", "version": VERIFIER_VERSION, "tests": tests}


def build_task(rng: random.Random, domain: str, difficulty: int) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be in [1, 5]")
    if domain == "math":
        prompt, verifier = _math_task(rng, difficulty)
    elif domain == "structured":
        prompt, verifier = _structured_task(rng, difficulty)
    else:
        prompt, verifier = _code_task(rng, difficulty)
    base = {
        "format_version": "1.0",
        "generator": GENERATOR_VERSION,
        "domain": domain,
        "difficulty": difficulty,
        "prompt": prompt,
        "verifier": verifier,
        "provenance": {"kind": "procedural", "generator": GENERATOR_VERSION},
    }
    return {"id": _task_id(base), **base}


def generate_tasks(
    *,
    seed: int,
    count: int,
    min_difficulty: int = 1,
    max_difficulty: int = 5,
    domains: tuple[str, ...] = DOMAINS,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 1 <= min_difficulty <= max_difficulty <= 5:
        raise ValueError("difficulty range must be within [1, 5]")
    if not domains or any(domain not in DOMAINS for domain in domains):
        raise ValueError("domains must be a non-empty subset of supported domains")

    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 100
    while len(tasks) < count and attempts < max_attempts:
        attempts += 1
        domain = domains[(len(tasks) + rng.randrange(len(domains))) % len(domains)]
        difficulty = rng.randint(min_difficulty, max_difficulty)
        task = build_task(rng, domain, difficulty)
        fingerprint = task["id"]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        task["generation"] = {"seed": seed, "ordinal": len(tasks), "attempt": attempts}
        tasks.append(task)
    if len(tasks) != count:
        raise RuntimeError(f"duplicate control exhausted after {attempts} attempts")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic self-improvement challenge tasks without external AI APIs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=5)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()
    tasks = generate_tasks(
        seed=args.seed,
        count=args.count,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        domains=tuple(args.domains),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(_canonical(task) + "\n")
    print(f"generated {len(tasks)} unique tasks -> {args.output}")


if __name__ == "__main__":
    main()
