#!/usr/bin/env python3
"""Build an offline student workbook from the maintained 617-question bank."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

import jsonschema

REPO = Path(__file__).resolve().parents[2]
BANK = REPO / "data/root_io/question_bank/root_questions_617.jsonl"
SCHEMA = REPO / "data/root_io/schema/question.schema.json"
OUTPUT = Path(__file__).resolve().parent


def canonical_topic(topic):
    return topic.removesuffix(".b")


def inline(value):
    """Escape source text before rendering its backtick-delimited code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(value, quote=True))


def load_records(bank=BANK):
    rows = [json.loads(line) for line in bank.read_text().splitlines() if line.strip()]
    validator = jsonschema.Draft202012Validator(
        json.loads(SCHEMA.read_text()), format_checker=jsonschema.FormatChecker())
    errors = []
    for row in rows:
        errors.extend(f"{row.get('id')}: {e.message}" for e in validator.iter_errors(row))
        for source in row.get("provenance", []):
            if not source["url"].startswith("https://"):
                errors.append(f"{row.get('id')}: source must use HTTPS")
    if len(rows) != 617:
        errors.append(f"Expected 617 records; got {len(rows)}")
    if {r["id"] for r in rows} != {f"root_knowledge_{i:03d}" for i in range(1, 618)}:
        errors.append("Missing or duplicate question IDs")
    if any(r["review_status"] == "rejected" for r in rows):
        errors.append("Rejected records must not be published")
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def render(rows):
    seen = set()
    cards = []
    for r in rows:
        topic = canonical_topic(r["topic"])
        role = "practice" if topic in seen else "core"
        seen.add(topic)
        sources = " · ".join(
            f'<a href="{html.escape(s["url"], quote=True)}" rel="noreferrer">{inline(s["title"])}</a>'
            for s in r["provenance"])
        context = f'<p class="context">{inline(r["context"])}</p>' if r.get("context") else ""
        cards.append(f'''<article id="{r['id']}" data-role="{role}" data-family="{topic.split('.')[0]}" data-level="{r['difficulty']}">
<p class="meta">{r['id']} · {inline(topic)} · {r['difficulty']} · {role}</p>
<h2>{inline(r['question'])}</h2>{context}
<details><summary>Show answer</summary><p>{inline(r['answer'])}</p>
<p class="sources">Further reading: {sources}</p></details></article>''')
    options = "".join(f'<option>{html.escape(t)}</option>' for t in sorted({r['topic'].split('.')[0] for r in rows}))
    return '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROOT and HEP — student workbook</title>
<style>
body{font:17px/1.55 system-ui,sans-serif;max-width:960px;margin:auto;padding:24px;color:#182438;background:#f7f9fc}
h1{line-height:1.2}h2{font-size:1.12rem}article{background:white;border:1px solid #c8d1df;border-radius:8px;margin:18px 0;padding:18px}
.meta,.sources{font-size:.85rem}.context{border-left:4px solid #546f99;padding:12px;background:#eef3fa}
code{overflow-wrap:anywhere}a{color:#14549b}summary{cursor:pointer;color:#14549b;font-weight:600}
input,select,button{font:inherit;padding:6px;margin:4px;max-width:95%}label{display:inline-block}
[hidden]{display:none!important}nav{border:1px solid #c8d1df;padding:12px}footer{margin-top:32px}
@media print{nav{display:none}article{break-inside:avoid}body{background:white}}
</style>
<header><h1>ROOT files and high-energy physics</h1>
<p>617 questions with an answer key · Student self-study edition · September 2026</p>
<p>Try each question before revealing the answer. Explain your reasoning and check the cited documentation.
Start with core questions; practice variants revisit a concept in another scenario and are not 617 independent skills.
Topics span file handling, event analysis, histograms, fitting and HEP interpretation.</p>
<p><strong>Scope:</strong> Text scenarios are hypothetical, not measurements from supplied files.
The accompanying <a href="README.md">lab guide</a> provides separate executable exercises.
Sources are further reading, not evidence for hypothetical file contents. External links require internet; the questions and answers work offline.</p>
<p>This is an editorially revised learning resource, not collaboration-approved analysis guidance or an expert-certified benchmark.
Advanced statistical answers depend on their stated assumptions. Report issues using the question ID and ROOT version.</p></header>
<nav aria-label="Question filters">
<label>Search <input id="search" type="search" placeholder="Question, answer or ID"></label>
<label>Topic <select id="family"><option value="">All topics</option>''' + options + '''</select></label>
<label>Level <select id="level"><option value="">All levels</option><option>introductory</option><option>intermediate</option><option>advanced</option></select></label>
<label>Set <select id="role"><option value="">All questions</option><option value="core">Core concepts</option><option value="practice">Practice variants</option></select></label>
<button id="reveal" type="button">Show visible answers</button><button id="hide" type="button">Hide all answers</button>
<p id="count" aria-live="polite">617 questions</p></nav>
<main>''' + "\n".join(cards) + '''</main>
<footer>Public answer key: do not use these public questions as a secret test set. No automatic RL reward verifier is supplied.</footer>
<script>
const cards=[...document.querySelectorAll('article')];
const controls=['search','family','level','role'].map(id=>document.getElementById(id));
function filter(){const [search,family,level,role]=controls.map(e=>e.value.toLowerCase());
 for(const card of cards)card.hidden=!(card.textContent.toLowerCase().includes(search)&&(!family||card.dataset.family===family)&&(!level||card.dataset.level===level)&&(!role||card.dataset.role===role));
 document.getElementById('count').textContent=cards.filter(c=>!c.hidden).length+' questions shown';}
controls.forEach(e=>e.addEventListener('input',filter));
document.getElementById('reveal').onclick=()=>cards.filter(c=>!c.hidden).forEach(c=>c.querySelector('details').open=true);
document.getElementById('hide').onclick=()=>cards.forEach(c=>c.querySelector('details').open=false);
</script></html>
'''


def build(output=OUTPUT):
    rows = load_records()
    output.mkdir(parents=True, exist_ok=True)
    document = render(rows)
    (output / "index.html").write_text(document, encoding="utf-8")
    answers = defaultdict(list)
    for r in rows:
        key = re.sub(r"[^a-z0-9]+", " ", r["answer"].lower()).strip()
        answers[key].append(r)
    repeats = [v for v in answers.values() if len(v) > 1]
    core = len({canonical_topic(r["topic"]) for r in rows})
    manifest = {
        "edition": "student-self-study-2026-09", "records": len(rows),
        "core_concepts": core, "practice_variants": len(rows)-core,
        "grouping": "First record per topic (excluding .b suffix) is core; this is not semantic deduplication.",
        "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "workbook_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "duplicate_answer_groups": len(repeats),
        "duplicate_answer_groups_crossing_legacy_splits": sum(len({r['split'] for r in g}) > 1 for g in repeats),
        "review": "Targeted editorial corrections; not independent expert certification. Machine-training review_status remains draft.",
        "training": "Legacy splits are not certified leakage-free. Public workbook is not a held-out evaluation. No RL verifier provided."
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(build(parser.parse_args().output), indent=2))
