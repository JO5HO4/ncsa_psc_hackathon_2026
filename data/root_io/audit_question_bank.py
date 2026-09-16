#!/usr/bin/env python3
"""Validate and summarize a ROOT knowledge question bank."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_SOURCE_HOSTS = {"root.cern", "uproot.readthedocs.io", "opendata.atlas.cern"}


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    import jsonschema

    schema_path = Path(__file__).with_name("schema") / "question.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.bank.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.expected_count is not None and len(records) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} records, found {len(records)}")

    ids: set[str] = set()
    questions: set[str] = set()
    answers: set[str] = set()
    group_splits: dict[str, str] = {}
    group_topics: set[tuple[str, str]] = set()
    for record in records:
        jsonschema.validate(record, schema)
        if record["id"] in ids:
            raise ValueError(f"Duplicate ID: {record['id']}")
        ids.add(record["id"])
        question_key = normalized(record["question"])
        if question_key in questions:
            raise ValueError(f"Duplicate normalized question: {record['question']}")
        questions.add(question_key)
        answer_key = normalized(record["answer"])
        if answer_key in answers:
            raise ValueError(f"Duplicate normalized answer: {record['id']}")
        answers.add(answer_key)
        previous = group_splits.setdefault(record["split_group"], record["split"])
        if previous != record["split"]:
            raise ValueError(f"Split leakage for group {record['split_group']}")
        group_topic = (record["split_group"], record["topic"])
        if group_topic in group_topics:
            raise ValueError(f"Repeated task type within one context: {group_topic}")
        group_topics.add(group_topic)
        for source in record["provenance"]:
            if urlparse(source["url"]).hostname not in ALLOWED_SOURCE_HOSTS:
                raise ValueError(f"Unapproved provenance host in {record['id']}: {source['url']}")

    summary = {
        "records": len(records),
        "splits": Counter(row["split"] for row in records),
        "difficulty": Counter(row["difficulty"] for row in records),
        "topic_families": Counter(row["topic"].split(".", 1)[0] for row in records),
        "review_status": Counter(row["review_status"] for row in records),
        "question_kinds": Counter(row["question_kind"] for row in records),
        "split_groups": len(group_splits),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
