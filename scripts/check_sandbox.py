"""Sanity: reference solutions must pass in the sandbox; an empty solution must fail."""
import time
from prunelab.codeeval import _pq, load_eval_sets, make_job
from prunelab.sandbox import run_jobs

items = load_eval_sets()
he = _pq("datasets--evalplus--humanevalplus/snapshots/*/data/*.parquet").set_index("task_id")
mb = _pq("datasets--evalplus--mbppplus/snapshots/*/data/*.parquet")
mb = {f"Mbpp/{r.task_id}": r.code for r in mb.itertuples()}
import json
from prunelab.common import DATA
kc = {}
p = DATA / "kodcode_heldout.jsonl"
if p.exists():
    kc = {json.loads(l)["question_id"]: json.loads(l)["solution"] for l in open(p)}
jobs, bad = [], []
for it in items:
    if it["set"] == "humaneval+":
        code = he.loc[it["id"]].prompt + he.loc[it["id"]].canonical_solution
    elif it["set"] == "mbpp+":
        code = mb[it["id"]]
    else:
        code = kc[it["id"]]
    jobs.append(make_job(it, code))
    bad.append(make_job(it, "pass"))
t = time.time()
res = run_jobs(jobs)
print(f"{time.time()-t:.0f}s")
by = {}
for j in jobs:
    s = j["id"].split("::")[0]; by.setdefault(s, [0, 0]); by[s][0] += 1; by[s][1] += res[j["id"]]["passed"]
print("reference pass:", by)
fails = [r for r in res.values() if not r["passed"]]
for r in fails[:5]: print("FAIL", r["id"], r["timeout"], r["tail"][-200:])
res2 = run_jobs(bad[:50])
print("empty solutions passing (should be 0):", sum(r["passed"] for r in res2.values()))
