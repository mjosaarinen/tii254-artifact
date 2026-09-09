#!/usr/bin/env python3
"""Reject proof holes and print the axioms of the paper-facing theorems."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
for path in sorted((ROOT / "ProjectedKernelRigidity").glob("*.lean")):
    text = path.read_text(encoding="utf-8")
    for forbidden in (r"\bsorry\b", r"\badmit\b", r"\bnative_decide\b"):
        if re.search(forbidden, text):
            raise SystemExit("forbidden proof mechanism in %s: %s" % (path, forbidden))

subprocess.run(["lake", "build"], cwd=ROOT, check=True)
completed = subprocess.run(
    ["lake", "env", "lean", "AxiomAudit.lean"],
    cwd=ROOT,
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(completed.stdout, end="")
if completed.returncode:
    raise SystemExit("Lean axiom audit failed")
allowed = {"propext", "Classical.choice", "Quot.sound"}
for body in re.findall(r"depends on axioms:\s*\[([^]]*)\]", completed.stdout, re.DOTALL):
    observed = {item.strip() for item in body.replace("\n", " ").split(",") if item.strip()}
    unexpected = observed - allowed
    if unexpected:
        raise SystemExit("unexpected axioms: " + ", ".join(sorted(unexpected)))
print("PASS: no proof holes; paper-facing axiom audit completed")
