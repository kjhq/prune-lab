"""Execute model-written code in a locked-down Docker container (no network, read-only root,
memory/pid caps, non-root user)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

IMAGE = "prunelab-sandbox"
RUNNER = Path(__file__).resolve().parents[2] / "docker" / "runner.py"


def run_jobs(jobs: list[dict], workers: int = 16, wall_timeout: int = 1800) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="prunelab-sbx-") as tmp:
        tmp = Path(tmp)
        (tmp / "jobs").mkdir()
        (tmp / "out").mkdir()
        (tmp / "out").chmod(0o777)
        with open(tmp / "jobs" / "manifest.jsonl", "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=4g", "--memory", "12g", "--cpus", "16", "--pids-limit", "2048",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{tmp / 'jobs'}:/jobs:ro", "-v", f"{tmp / 'out'}:/out:rw", "-v", f"{RUNNER}:/runner.py:ro",
            IMAGE, "python", "/runner.py", str(workers),
        ]
        subprocess.run(cmd, check=True, timeout=wall_timeout, capture_output=True)
        res = [json.loads(l) for l in open(tmp / "out" / "results.jsonl")]
    return {r["id"]: r for r in res}
