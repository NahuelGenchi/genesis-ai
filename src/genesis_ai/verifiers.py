from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")
MAX_EXPRESSION_CHARS = 256
MAX_AST_NODES = 64
MAX_ABS_VALUE = 10**12
MAX_EXPONENT = 8
VERIFIER_VERSION = "deterministic-v1"


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(passed: bool, score: float, reason: str, **details: Any) -> VerificationResult:
    return VerificationResult(passed=passed, score=score, reason=reason, details=details)


def verify_integer(spec: dict[str, Any], answer: str) -> VerificationResult:
    stripped = answer.strip()
    if not INTEGER_RE.fullmatch(stripped):
        return _result(False, 0.0, "invalid_integer")
    expected = spec.get("expected")
    if not isinstance(expected, int) or isinstance(expected, bool):
        raise ValueError("integer_exact verifier requires integer expected")
    actual = int(stripped)
    if actual != expected:
        return _result(False, 0.0, "wrong_answer", expected=expected, actual=actual)
    return _result(True, 1.0, "correct")


def verify_json(spec: dict[str, Any], answer: str) -> VerificationResult:
    try:
        actual = json.loads(answer)
    except json.JSONDecodeError as exc:
        return _result(False, 0.0, "invalid_json", error=exc.msg)
    if "expected" not in spec:
        raise ValueError("json_exact verifier requires expected")
    expected = spec["expected"]
    if actual != expected:
        return _result(False, 0.0, "wrong_answer", expected=expected, actual=actual)
    return _result(True, 1.0, "correct")


class RestrictedExpressionError(ValueError):
    pass


def _bounded(value: int) -> int:
    if abs(value) > MAX_ABS_VALUE:
        raise RestrictedExpressionError("magnitude_limit")
    return value


def _eval_node(node: ast.AST, variables: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise RestrictedExpressionError("integer_constants_only")
        return _bounded(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise RestrictedExpressionError("unknown_variable")
        return _bounded(variables[node.id])
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return _bounded(-value)
        raise RestrictedExpressionError("unsafe_operator")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return _bounded(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded(left * right)
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise RestrictedExpressionError("division_by_zero")
            return _bounded(left // right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise RestrictedExpressionError("division_by_zero")
            return _bounded(left % right)
        if isinstance(node.op, ast.Pow):
            if right < 0 or right > MAX_EXPONENT:
                raise RestrictedExpressionError("exponent_limit")
            return _bounded(left**right)
        raise RestrictedExpressionError("unsafe_operator")
    raise RestrictedExpressionError("unsafe_syntax")


def parse_restricted_expression(expression: str) -> ast.Expression:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise RestrictedExpressionError("expression_too_long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise RestrictedExpressionError("invalid_syntax") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise RestrictedExpressionError("ast_too_large")
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    if any(not isinstance(node, allowed) for node in nodes):
        raise RestrictedExpressionError("unsafe_syntax")
    return tree


def verify_restricted_expression(spec: dict[str, Any], answer: str) -> VerificationResult:
    tests = spec.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("restricted_expression verifier requires tests")
    try:
        tree = parse_restricted_expression(answer.strip())
    except RestrictedExpressionError as exc:
        return _result(False, 0.0, str(exc))

    passed = 0
    failures: list[dict[str, Any]] = []
    for index, test in enumerate(tests):
        if not isinstance(test, dict) or not isinstance(test.get("variables"), dict):
            raise ValueError("restricted_expression test is invalid")
        variables = test["variables"]
        expected = test.get("expected")
        if (
            not isinstance(expected, int)
            or isinstance(expected, bool)
            or any(not isinstance(name, str) or not isinstance(value, int) or isinstance(value, bool) for name, value in variables.items())
        ):
            raise ValueError("restricted_expression test values must be integers")
        try:
            actual = _eval_node(tree, variables)
        except RestrictedExpressionError as exc:
            return _result(False, 0.0, str(exc), test_index=index)
        if actual == expected:
            passed += 1
        else:
            failures.append({"test_index": index, "expected": expected, "actual": actual})

    score = passed / len(tests)
    if score == 1.0:
        return _result(True, 1.0, "correct", tests=len(tests))
    return _result(False, score, "wrong_answer", passed_tests=passed, total_tests=len(tests), failures=failures[:3])


def verify_task(task: dict[str, Any], answer: str) -> VerificationResult:
    verifier = task.get("verifier")
    if not isinstance(verifier, dict):
        raise ValueError("task verifier must be an object")
    if verifier.get("version") != VERIFIER_VERSION:
        raise ValueError("unsupported verifier version")
    kind = verifier.get("kind")
    if kind == "integer_exact":
        return verify_integer(verifier, answer)
    if kind == "json_exact":
        return verify_json(verifier, answer)
    if kind == "restricted_expression":
        return verify_restricted_expression(verifier, answer)
    raise ValueError(f"unsupported verifier kind: {kind}")
