from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent
DISPATCH = json.loads((PAYLOAD / "dispatch.json").read_text(encoding="utf-8"))
SOURCE = Path("/kaggle/working/genesis-ai-source")
OUTPUT = Path("/kaggle/working/genesis-output")


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
print(f"Genesis accelerator output: {OUTPUT}")
