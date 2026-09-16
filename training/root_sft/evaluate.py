#!/usr/bin/env python3
"""Prepare blinded paired review and summarize human ROOT-answer scores, not RL rewards."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.root_sft.prepare import digest, jsonl, prompt

CRITERIA = ("correctness", "completeness", "assumptions", "clarity")


def records(path, completions=False):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if completions and row.get("record_type") == "metadata":
            continue
        if completions and row.get("record_type") != "completion":
            raise ValueError("Expected inference completion records")
        if row["id"] in result:
            raise ValueError(f"Duplicate ID {row['id']}")
        result[row["id"]] = row
    return result


def prepare(references, before, after, output, expected_limit=None):
    refs, pre, post = records(references), records(before, True), records(after, True)
    expected = set(list(refs)[:expected_limit]) if expected_limit is not None else set(refs)
    if not expected or set(pre) != expected or set(post) != expected:
        raise ValueError("Missing, unexpected or mismatched prediction IDs; use --limit only for deliberate smoke subsets")
    if output.exists():
        raise FileExistsError(f"Refusing existing review directory: {output}")
    rng, paired, mapping = random.Random(20260908), [], {}
    for id_ in sorted(pre):
        r = refs[id_]
        if pre[id_]["prompt"] != prompt(r) or post[id_]["prompt"] != prompt(r):
            raise ValueError(f"Prompt mismatch for {id_}")
        order = ["before", "after"]
        rng.shuffle(order)
        mapping[id_] = dict(zip(("A", "B"), order))
        values = {"before": pre[id_]["output"], "after": post[id_]["output"]}
        paired.append({"id": id_, "category": r["category"], "topic": r["topic"],
            "difficulty": r["difficulty"], "prompt": prompt(r), "reference": r["answer"],
            "required_facts": r["required_facts"], "provenance": r["provenance"],
            "A": values[order[0]], "B": values[order[1]],
            "scores": {s: {**dict.fromkeys(CRITERIA), "critical_error": None, "notes": ""} for s in ("A", "B")}})
    output.mkdir(parents=True)
    jsonl(output/"paired_review.jsonl", paired)
    (output/"private_mapping.json").write_text(json.dumps(mapping, indent=2)+"\n")
    (output/"inputs.json").write_text(json.dumps({"references_sha256": digest(references),
        "before_sha256": digest(before), "after_sha256": digest(after), "n": len(paired),
        "subset_only": expected_limit is not None}, indent=2)+"\n")
    return paired


def report(scored, mapping_file):
    rows = records(scored)
    mapping = json.loads(mapping_file.read_text())
    if not rows or set(rows) != set(mapping):
        raise ValueError("Review and mapping IDs must agree exactly")
    buckets = {"before": defaultdict(list), "after": defaultdict(list)}
    deltas = []
    for id_, r in rows.items():
        totals = {}
        for slot in ("A", "B"):
            s = r["scores"][slot]
            if any(type(s.get(k)) is not int or not 0 <= s[k] <= 2 for k in CRITERIA):
                raise ValueError(f"{id_}/{slot}: all four human scores must be integers 0..2")
            if type(s.get("critical_error")) is not bool:
                raise ValueError(f"{id_}/{slot}: critical_error must be true or false")
            total = 0 if s["critical_error"] else sum(s[k] for k in CRITERIA)/8
            model = mapping[id_][slot]
            if model not in buckets:
                raise ValueError("Invalid model mapping")
            totals[model] = total
            for key in ("overall", "category:"+r["category"], "difficulty:"+r["difficulty"], "topic:"+r["topic"]):
                buckets[model][key].append(total)
        if set(totals) != {"before", "after"}:
            raise ValueError("Each pair must contain before and after exactly once")
        deltas.append(totals["after"]-totals["before"])
    summary = {m: {k: {"n": len(v), "mean": statistics.mean(v)} for k, v in b.items()} for m, b in buckets.items()}
    for model in buckets:
        summary[model]["macro_category_mean"] = statistics.mean(v["mean"] for k, v in summary[model].items() if k.startswith("category:"))
    return {"n": len(rows), "scores": summary, "paired_mean_delta": statistics.mean(deltas),
        "improved": sum(d > 0 for d in deltas), "tied": sum(d == 0 for d in deltas),
        "regressed": sum(d < 0 for d in deltas),
        "warning": "Human rubric scores on a small public knowledge set; not an executable ROOT benchmark or proof of general HEP improvement."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    for key in ("references", "before", "after", "output"):
        a.add_argument("--"+key, type=Path, required=True)
    a.add_argument("--limit", type=int)
    b = sub.add_parser("report")
    for key in ("scored", "mapping", "output"):
        b.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        if args.limit is not None and args.limit < 1:
            p.error("--limit must be positive")
        prepare(args.references, args.before, args.after, args.output, args.limit)
    else:
        result = report(args.scored, args.mapping)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
