#!/usr/bin/env python3
"""Expand the reviewed pilot design into a 500-record ROOT/HEP draft bank."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PILOT = HERE / "question_bank/pilot.jsonl"
OUTPUT = HERE / "question_bank/root_questions_500.jsonl"
TREE_SOURCE = {"title": "ROOT trees manual", "url": "https://root.cern/manual/trees/"}

# channel, collection, count, pt, eta, phi, quality, histogram
CHANNELS = [
    ("diphoton", "photon", "photon_n", "photon_pt", "photon_eta", "photon_phi", "photon_isTightID", "h_myy"),
    ("dielectron", "electron", "electron_n", "electron_pt", "electron_eta", "electron_phi", "electron_charge", "h_mee"),
    ("dimuon", "muon", "muon_n", "muon_pt", "muon_eta", "muon_phi", "muon_isolated", "h_mumu"),
    ("single_lepton_top", "jet", "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_btag", "h_top_mass"),
    ("four_lepton", "lepton", "lepton_n", "lepton_pt", "lepton_eta", "lepton_phi", "lepton_flavor", "h_m4l"),
    ("jets_met", "jet", "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_jvt", "h_met"),
    ("tau_lepton", "tau", "tau_n", "tau_pt", "tau_eta", "tau_phi", "tau_isTight", "h_tau_pt"),
    ("b_physics", "track", "track_n", "track_pt", "track_eta", "track_phi", "track_charge", "h_vertex_mass"),
    ("vbf", "jet", "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_jvt", "h_mjj"),
    ("all_hadronic_top", "largeRjet", "largeRjet_n", "largeRjet_pt", "largeRjet_eta", "largeRjet_phi", "largeRjet_topTag", "h_largeRjet_mass"),
]
SAMPLES = [
    ("collision_data", "data", "run_number", None),
    ("signal_simulation", "signal MC", "mc_channel_number", "event_weight"),
    ("background_simulation", "background MC", "mc_channel_number", "event_weight"),
]


def split_map(groups: list[str]) -> dict[str, str]:
    ranked = sorted(groups, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    return {group: "train" if i < 24 else "validation" if i < 27 else "test" for i, group in enumerate(ranked)}


def cases(s: dict[str, object]):
    label, tree, entries = s["label"], s["tree"], s["entries"]
    c, n, pt, eta, phi, quality, hist = (s[k] for k in ("collection", "count", "pt", "eta", "phi", "quality", "hist"))
    weight, sample = s["weight"], s["sample"]
    inventory = f"File `{label}.root` contains TTree `{tree}` with {entries} entries and TH1D `{hist}`."
    branches = f"Tree metadata lists scalar `{n}`, jagged `{pt}`, `{eta}`, `{phi}`, and `{quality}`, plus scalar `{s['identifier']}`" + (f" and scalar `{weight}`." if weight else ".")
    return [
        ("inventory.event_container", "introductory", inventory, f"Which object should be opened to inspect event-level records in `{label}.root`?", f"Open `{tree}`. It is the TTree in this inventory; `{hist}` is a histogram rather than the event-level columnar container.", [f"Select TTree {tree}"]),
        ("inventory.entry_count", "introductory", inventory, f"How many stored rows should an inspection report for `{tree}` in this scenario?", f"Report {entries} entries for `{tree}`. Calling them physics events is appropriate only if the fixture documentation confirms that this is an event-level tree.", [f"Report {entries} entries", "Do not infer row semantics without documentation"]),
        ("schema.multiplicity", "introductory", branches, f"Which branch records the per-entry {c} multiplicity?", f"Use `{n}` as the scalar {c} multiplicity branch, then cross-check it against the lengths of the jagged {c} collections.", [f"Identify {n}", "Cross-check jagged lengths"]),
        ("schema.jagged", "introductory", branches, f"Which listed branches require variable-length array handling?", f"`{pt}`, `{eta}`, `{phi}`, and `{quality}` are jagged in this schema because each entry can contain a different number of {c}s.", [f"Identify {pt}, {eta}, {phi}, and {quality} as jagged"]),
        ("schema.identifiers", "intermediate", branches, f"Which scalar branch distinguishes the sample or acquisition context for this {sample} file?", f"The listed scalar identifier is `{s['identifier']}`. Its precise interpretation still comes from the fixture documentation.", [f"Identify {s['identifier']}", "Require documentation for semantics"]),
        ("analysis.minimal_columns", "intermediate", branches, f"What is the minimal listed branch set for studying the {c} transverse-momentum distribution versus pseudorapidity?", f"Read `{pt}` and `{eta}`; also read `{n}` to validate per-entry collection lengths. `{phi}` and `{quality}` are not required for that narrowly stated distribution.", [f"Read {pt} and {eta}", f"Use {n} for consistency"]),
        ("analysis.safe_indexing", "intermediate", branches, f"What check is required before accessing the leading two values in `{pt}`?", f"Require `{n} >= 2` and verify the `{pt}` collection actually has at least two elements for that entry before indexing positions 0 and 1.", [f"Require {n} >= 2", "Verify actual collection length"]),
        ("analysis.selection", "intermediate", branches, f"How should a two-{c} preselection be expressed conceptually without reading unrelated columns?", f"Read `{n}` first and retain entries with `{n} >= 2`; load additional {c} branches only when later cuts need them.", [f"Select with {n} >= 2", "Avoid unrelated columns"]),
        ("histogram.crosscheck", "intermediate", inventory + " " + branches, f"How can `{hist}` be used when validating a newly calculated distribution?", f"Inspect `{hist}` binning, flow bins, contents, and uncertainties, then compare it with a calculation made under the documented selection and weighting. Matching names alone is not sufficient.", [f"Inspect {hist} metadata and values", "Match selection and weighting"]),
        ("weights.sample_type", "intermediate", branches, f"Should an event-weight branch be used for this {sample} scenario?", (f"Yes, `{weight}` is listed, but its factors and normalization must be verified before use." if weight else "No event-weight branch is listed for this collision-data schema; do not invent or apply a Monte Carlo weight."), [f"Respect the listed weighting schema for {sample}"]),
        ("semantics.units", "introductory", branches, f"May the unit of `{pt}` be asserted from this branch inventory alone?", f"No. `{pt}` suggests transverse momentum, but its unit and calibration must be taken from fixture documentation or explicit metadata.", ["Do not infer units from a branch name"]),
        ("tools.sequence", "introductory", inventory, f"What bounded tool sequence should an agent use to summarize `{label}.root`?", f"First list ROOT objects, then describe `{tree}` with targeted branch patterns, and summarize only branches needed by the question. Avoid arbitrary code and full-file reads.", ["Inventory before payload reads", "Use targeted bounded reads"]),
        ("performance.chunking", "intermediate", inventory + f" The requested study needs `{pt}` across all rows.", f"How should `{pt}` be processed if all {entries} entries do not fit comfortably in memory?", f"Read `{pt}` in bounded entry chunks and aggregate partial results. Keep other branches disabled unless the computation needs them.", ["Use bounded entry chunks", "Select only needed branches"]),
        ("validation.collection_lengths", "advanced", branches, f"What consistency test can detect malformed {c} collections?", f"For every inspected entry, compare `{n}` with the lengths of `{pt}`, `{eta}`, `{phi}`, and `{quality}`. Report disagreements rather than silently truncating arrays.", [f"Compare {n} to all collection lengths", "Report mismatches"]),
        ("reproducibility.fixture", "intermediate", inventory + f" The file belongs to the `{label}` fixture group.", "What must be recorded before this scenario becomes a reviewed training example?", f"Record the exact checksum, immutable source revision, object path `{tree}`, actual tool observations, and the `{label}` split group.", ["Record checksum and revision", f"Preserve split group {label}"]),
    ]


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "build_knowledge_pilot.py")], check=True)
    records = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines() if line]
    scenarios = []
    for channel_index, channel in enumerate(CHANNELS):
        channel_name, collection, count, pt, eta, phi, quality, hist = channel
        for sample_index, (sample_name, sample, identifier, weight) in enumerate(SAMPLES):
            label = f"{channel_name}_{sample_name}"
            scenarios.append({"label": label, "tree": f"events_{channel_name}", "entries": 12000 + 7919 * channel_index + 1237 * sample_index, "collection": collection, "count": count, "pt": pt, "eta": eta, "phi": phi, "quality": quality, "hist": hist, "sample": sample, "identifier": identifier, "weight": weight})
    splits = split_map([s["label"] for s in scenarios])
    index = 51
    for scenario in scenarios:
        for topic, difficulty, context, question, answer, facts in cases(scenario):
            label = scenario["label"]
            scoped_question = f"In the `{label}` scenario, {question[0].lower()}{question[1:]}"
            scoped_answer = f"For the `{label}` scenario, {answer[0].lower()}{answer[1:]}"
            records.append({"schema_version": "v0.1", "id": f"root_knowledge_{index:03d}", "question_kind": "contextual_analysis", "topic": topic, "difficulty": difficulty, "split": splits[label], "context": context, "question": scoped_question, "answer": scoped_answer, "required_facts": facts, "forbidden_claims": [], "provenance": [TREE_SOURCE], "training_use": ["sft"], "rl_eligible": False, "split_group": label, "review_status": "draft"})
            index += 1
    if len(records) != 500:
        raise RuntimeError(f"Expected 500 records, found {len(records)}")
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    print(f"Wrote {len(records)} questions to {OUTPUT}")


if __name__ == "__main__":
    main()
