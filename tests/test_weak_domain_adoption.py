import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.weak_domain_adoption import apply_adoption


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation(code, math, structured):
    return {
        "domains": {
            "code": {"exact_accuracy": code},
            "math": {"exact_accuracy": math},
            "structured": {"exact_accuracy": structured},
        }
    }


class WeakDomainAdoptionTest(unittest.TestCase):
    def _fixture(self, root: Path, *, promoted: bool, stale: bool = False):
        incumbent = root / "incumbent.pt"
        incumbent.write_bytes(b"incumbent")
        baseline_sha = sha(incumbent)
        candidate = root / "candidate.pt"
        candidate.write_bytes(b"candidate")
        candidate_sha = sha(candidate)
        state = root / "state.json"
        state.write_text(
            json.dumps(
                {
                    "state_version": "autonomous-state-v1",
                    "cash_compute_cost_usd": 0.0,
                    "incumbent_checkpoint": incumbent.as_posix(),
                    "incumbent_gci_v1": 31.6667,
                    "autonomy_status": "research_hold",
                    "circuit_breaker": {"active": True, "reason": "exhausted"},
                }
            ),
            encoding="utf-8",
        )
        gate = root / "gate.json"
        gate.write_text(
            json.dumps(
                {
                    "promoted": promoted,
                    "decision": "promote" if promoted else "reject",
                    "baseline_checkpoint_sha256": baseline_sha,
                    "candidate_checkpoint_sha256": candidate_sha,
                }
            ),
            encoding="utf-8",
        )
        candidate_eval = root / "candidate-eval.json"
        candidate_eval.write_text(json.dumps(evaluation(0.95, 0.15, 0.10)), encoding="utf-8")
        if stale:
            incumbent.write_bytes(b"newer-incumbent")
        return state, baseline_sha, gate, candidate_eval, candidate

    def test_promoted_fresh_candidate_replaces_incumbent_and_wakes_autonomy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, baseline_sha, gate, candidate_eval, candidate = self._fixture(root, promoted=True)
            destination = root / "promoted.pt"
            adoption = apply_adoption(
                state_path=state,
                baseline_sha256=baseline_sha,
                gate_path=gate,
                candidate_evaluation_path=candidate_eval,
                candidate_checkpoint=candidate,
                destination_checkpoint=destination,
                output_path=root / "adoption.json",
            )
            self.assertTrue(adoption["adopted"])
            persisted = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["incumbent_checkpoint"], destination.as_posix())
            self.assertEqual(persisted["autonomy_status"], "running")
            self.assertFalse(persisted["circuit_breaker"]["active"])
            self.assertAlmostEqual(persisted["incumbent_gci_v1"], 40.0)

    def test_stale_incumbent_fails_closed_even_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, baseline_sha, gate, candidate_eval, candidate = self._fixture(root, promoted=True, stale=True)
            destination = root / "promoted.pt"
            adoption = apply_adoption(
                state_path=state,
                baseline_sha256=baseline_sha,
                gate_path=gate,
                candidate_evaluation_path=candidate_eval,
                candidate_checkpoint=candidate,
                destination_checkpoint=destination,
                output_path=root / "adoption.json",
            )
            self.assertFalse(adoption["adopted"])
            self.assertFalse(destination.exists())
            persisted = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["autonomy_status"], "research_hold")

    def test_rejected_candidate_never_changes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, baseline_sha, gate, candidate_eval, candidate = self._fixture(root, promoted=False)
            before = state.read_text(encoding="utf-8")
            adoption = apply_adoption(
                state_path=state,
                baseline_sha256=baseline_sha,
                gate_path=gate,
                candidate_evaluation_path=candidate_eval,
                candidate_checkpoint=candidate,
                destination_checkpoint=root / "promoted.pt",
                output_path=root / "adoption.json",
            )
            self.assertFalse(adoption["adopted"])
            self.assertEqual(state.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
