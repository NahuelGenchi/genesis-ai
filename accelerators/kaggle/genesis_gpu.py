from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent
DISPATCH = json.loads((PAYLOAD / "dispatch.json").read_text(encoding="utf-8"))
SOURCE = Path("/kaggle/working/genesis-ai-source")
OUTPUT = Path("/kaggle/working/genesis-output")
CORPUS = SOURCE / "runs/accelerator-public-corpus"
REBUILT_TOKENIZER = SOURCE / "runs/accelerator-rebuilt-tokenizer.json"
LOCK_COPY = SOURCE / "runs/accelerator-bootstrap-lock.json"


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


if SOURCE.exists():
    run("rm", "-rf", str(SOURCE))
SOURCE.mkdir(parents=True)
run("git", "init", str(SOURCE))
run("git", "-C", str(SOURCE), "remote", "add", "origin", DISPATCH["repo_url"])
run("git", "-C", str(SOURCE), "fetch", "--depth=1", "origin", DISPATCH["ref"])
run("git", "-C", str(SOURCE), "checkout", "--detach", "FETCH_HEAD")
actual_ref = subprocess.check_output(
    ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
).strip()
if actual_ref != DISPATCH["ref"]:
    raise RuntimeError(f"repository ref mismatch: expected {DISPATCH['ref']}, got {actual_ref}")

run(sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-e", str(SOURCE))

# Accelerator jobs are required to train on the same provenance-locked public corpus
# used by canonical remote workflows. Rebuild it inside the ephemeral Kaggle runtime.
LOCK_COPY.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(SOURCE / "data/bootstrap-tokenizer-lock.json", LOCK_COPY)
run(
    sys.executable,
    "-m",
    "genesis_ai.bootstrap_corpus",
    "--catalog",
    "data/bootstrap-tokenizer-sources.json",
    "--workspace",
    str(CORPUS.relative_to(SOURCE)),
    "--output",
    str(REBUILT_TOKENIZER.relative_to(SOURCE)),
    "--lock",
    str(LOCK_COPY.relative_to(SOURCE)),
    "--vocab-size",
    "512",
    cwd=SOURCE,
)
if LOCK_COPY.read_bytes() != (SOURCE / "data/bootstrap-tokenizer-lock.json").read_bytes():
    raise RuntimeError("public source lock changed during Kaggle rebuild")
if REBUILT_TOKENIZER.read_bytes() != (SOURCE / "tokenizers/genesis-v0.json").read_bytes():
    raise RuntimeError("rebuilt tokenizer differs from the committed tokenizer")

OUTPUT.mkdir(parents=True, exist_ok=True)
run(
    sys.executable,
    "-m",
    "genesis_ai.accelerator_job",
    "run",
    "--job",
    str(PAYLOAD / "job.json"),
    "--cpu-summary",
    str(PAYLOAD / "cpu-farm-summary.json"),
    "--platform",
    "kaggle",
    "--output-dir",
    str(OUTPUT),
    cwd=SOURCE,
)

record = json.loads((OUTPUT / "accelerator-record.json").read_text(encoding="utf-8"))
record["source_commit"] = actual_ref
record["cpu_farm_run_id"] = int(DISPATCH["cpu_farm_run_id"])
record["cash_compute_cost_usd"] = 0.0
(OUTPUT / "accelerator-record.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"Genesis accelerator output: {OUTPUT}")
