"""Runs inside the sandbox container (no network, read-only root).

Reads /jobs/manifest.jsonl: {"id", "files": {name: text}, "cmd": [...], "timeout": s}.
Writes /out/results.jsonl: {"id", "passed", "rc", "timeout", "tail"}.
"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def run(job):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        for name, text in job["files"].items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        try:
            p = subprocess.run(
                job["cmd"], cwd=d, capture_output=True, text=True, timeout=job.get("timeout", 20),
                env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1", "HOME": d},
                stdin=subprocess.DEVNULL,
            )
            return {"id": job["id"], "passed": p.returncode == 0, "rc": p.returncode, "timeout": False,
                    "tail": (p.stdout + p.stderr)[-400:]}
        except subprocess.TimeoutExpired:
            return {"id": job["id"], "passed": False, "rc": None, "timeout": True, "tail": ""}


def main():
    jobs = [json.loads(l) for l in open("/jobs/manifest.jsonl")]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with ThreadPoolExecutor(workers) as ex, open("/out/results.jsonl", "w") as out:
        for r in ex.map(run, jobs):
            out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
