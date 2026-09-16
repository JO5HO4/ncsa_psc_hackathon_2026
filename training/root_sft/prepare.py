#!/usr/bin/env python3
"""Prepare conservative, topic-capped ROOT knowledge SFT data for existing verl."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from students.root_io.build_release import BANK, load_records

SYSTEM = "You are a careful ROOT and high-energy-physics tutor. Answer the question clearly. State relevant assumptions and do not invent file contents, units, or analysis results."
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topic(row):
    return row["topic"].lower().removesuffix(".b")


def category(row):
    family = topic(row).split(".")[0]
    if family == "stats":
        return "statistics"
    if family == "hep":
        return "hep_interpretation"
    if family in {"histogram", "histograms", "graphics"}:
        return "histograms_plotting"
    if family in {"io", "files", "inventory", "reproducibility", "tools"}:
        return "files_io"
    return "trees_analysis"


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def components(rows):
    """Transitive closure BEFORE sampling; discarded siblings still link groups."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        keys = [("topic", topic(r)), ("scenario", r["split_group"]),
                ("answer", normalized(r["answer"])), ("question", normalized(r["question"]))]
        for key in keys:
            if key in seen:
                a, b = find(i), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return {min(r["id"] for r in group): group for group in groups.values()}


def select_balanced(rows, cap, seed):
    """No oversampling: distinct answers, topic cap, diversity within each topic."""
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        by_topic[topic(row)].append(row)
    selected, reasons, answer_seen = [], {}, set()
    for key in sorted(by_topic):
        candidates = by_topic[key][:]
        rng.shuffle(candidates)
        levels, kinds = Counter(), Counter()
        chosen = 0
        while candidates:
            candidates.sort(key=lambda r: (levels[r["difficulty"]], kinds[r["question_kind"]]))
            r = candidates.pop(0)
            a = normalized(r["answer"])
            if a in answer_seen:
                reasons[r["id"]] = "duplicate_answer"
            elif chosen >= cap:
                reasons[r["id"]] = "topic_cap"
            else:
                selected.append(r)
                answer_seen.add(a)
                levels[r["difficulty"]] += 1
                kinds[r["question_kind"]] += 1
                chosen += 1
    return selected, reasons


def features(rows):
    counts = Counter({"total": len(rows)})
    for r in rows:
        counts["category:"+category(r)] += 1
        counts["difficulty:"+r["difficulty"]] += 1
        counts["kind:"+r["question_kind"]] += 1
    return counts


def assign_splits(groups, selected, seed):
    ids = {r["id"] for r in selected}
    active = {g: [r for r in rs if r["id"] in ids] for g, rs in groups.items()}
    totals = features(selected)
    counts = {s: Counter() for s in RATIOS}
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(active[g]))
    assigned = {}
    for g in order:
        f = features(active[g])
        # Large connected scenario groups cannot fairly fit a small holdout.
        allowed = ["train"] if len(active[g]) > totals["total"] * 0.15 else list(RATIOS)

        def cost(split):
            result = 0.
            for s, ratio in RATIOS.items():
                for key, total in totals.items():
                    value = counts[s][key] + (f[key] if s == split else 0)
                    weight = 4 if key == "total" else 1
                    result += weight * (value-ratio*total)**2 / max(total, 1)
            return result

        split = min(allowed, key=cost)
        assigned[g] = split
        counts[split].update(f)
    return assigned


def prompt(row):
    return (row.get("context", "") + "\n\n" + row["question"]).strip()


def chat(row, group, split):
    return {"id": row["id"], "logical_task_id": row["id"],
            "dataset_family": "root_io", "modality": "direct_qa", "harness": "none",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt(row)},
                         {"role": "assistant", "content": row["answer"]}],
            # Existing repository adapter explicitly decodes this representation.
            "tools": "[]", "topic": topic(row), "category": category(row),
            "difficulty": row["difficulty"], "question_kind": row["question_kind"],
            "split": split, "split_group": group,
            "provenance_json": json.dumps(row["provenance"], sort_keys=True),
            "source_review_status": row["review_status"]}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows), encoding="utf-8")


def build(bank, output, *, allow_draft=False, cap=2, seed=20260908):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if cap < 1:
        raise ValueError("topic cap must be positive")
    rows = sorted(load_records(bank), key=lambda r: r["id"])
    if not allow_draft and any(r["review_status"] != "reviewed" for r in rows):
        raise ValueError("Draft data: use --allow-draft for an explicitly experimental run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty release directory: {output}")
    groups = components(rows)
    selected, reasons = select_balanced(rows, cap, seed)
    assigned = assign_splits(groups, selected, seed)
    membership = {r["id"]: g for g, rs in groups.items() for r in rs}
    splits = {s: sorted([r for r in selected if assigned[membership[r['id']]] == s], key=lambda r: r['id']) for s in RATIOS}
    if any(not rs for rs in splits.values()):
        raise ValueError("Cannot construct nonempty independent splits with this grouping")
    output.mkdir(parents=True, exist_ok=True)
    for split, rs in splits.items():
        chats = [chat(r, membership[r['id']], split) for r in rs]
        pq.write_table(pa.Table.from_pylist(chats), output/f"{split}.parquet")
        jsonl(output/f"{split}.jsonl", chats)
        if split != "train":
            jsonl(output/f"{split}_prompts.jsonl", [{"id": r['id'], "prompt": prompt(r)} for r in rs])
            jsonl(output/f"{split}_references.jsonl", [dict(r, category=category(r), evaluation_split=split) for r in rs])
    selected_ids = {r['id'] for r in selected}
    jsonl(output/"assignments.jsonl", [{"id": r['id'], "group": membership[r['id']],
        "split": assigned[membership[r['id']]], "legacy_split": r['split'],
        "selected": r['id'] in selected_ids, "reason": reasons.get(r['id'], "selected"),
        "topic": topic(r), "category": category(r)} for r in rows])
    manifest = {"version": "root-knowledge-sft-v1", "source_sha256": digest(bank),
        "source_records": len(rows), "selected_records": len(selected), "seed": seed,
        "topic_cap": cap, "system_prompt": SYSTEM,
        "review_policy": "experimental draft accepted" if allow_draft else "reviewed only",
        "grouping": "Transitive union of canonical topic, original split_group, normalized exact answer and question; before selection",
        "balance_policy": "Global exact-answer deduplication; at most two per topic by default; prefer varied difficulty/kind. No oversampling or artificial answers.",
        "requested_ratios": RATIOS, "counts": {s: dict(features(rs)) for s, rs in splits.items()},
        "excluded": dict(Counter(reasons.values())),
        "groups": [{"id": g, "source_count": len(rs), "selected_count": sum(r['id'] in selected_ids for r in rs), "split": assigned[g]} for g, rs in sorted(groups.items())],
        "limitations": ["Grouping detects known links, not all semantic similarity or pretraining contamination.",
            "Large connected teaching-scenario groups can be train-only; holdouts may lack contextual tasks.",
            "Category and difficulty proportions follow available distinct concepts, not equal quotas.",
            "Public knowledge answers do not measure executable tool competence. No RL rewards."]}
    manifest["files_sha256"] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--topic-cap", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260908)
    args = p.parse_args()
    report = build(args.bank, args.output, allow_draft=args.allow_draft, cap=args.topic_cap, seed=args.seed)
    print(json.dumps({k: report[k] for k in ("source_records", "selected_records", "counts", "excluded")}, indent=2))
