#!/usr/bin/env python3
"""Generate OpenAI-compatible CBORG chat completions from JSONL prompts."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def records(path: Path, prompt_field: str, id_field: str) -> list[dict[str, str]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row.get(prompt_field), str):
            raise ValueError(f"missing string {prompt_field} in {row.get('id', '<unknown>')}")
        output.append({"id": str(row[id_field]), "prompt": row[prompt_field]})
    return output


def request_completion(base_url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="CBORG model name, e.g. the GPT-OSS-120B name returned by /model/info")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-field", default="question")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--requests-per-minute", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3, help="retries after transient API failures")
    parser.add_argument("--resume", action="store_true", help="append only IDs with no successful prior completion")
    parser.add_argument("--base-url", default="https://api.cborg.lbl.gov")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.requests_per_minute <= 0:
        raise ValueError("--requests-per-minute must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    api_key = os.environ.get("CBORG_API_KEY")
    if not api_key:
        raise EnvironmentError("CBORG_API_KEY is unset; source ~/.apikeys.sh before running")
    prompt_rows = records(args.prompts, args.prompt_field, args.id_field)
    if args.limit is not None:
        prompt_rows = prompt_rows[:args.limit]
    completed_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            prior = json.loads(line)
            if prior.get("record_type") == "completion" and isinstance(prior.get("output"), str) and prior["output"]:
                completed_ids.add(str(prior["id"]))
        prompt_rows = [row for row in prompt_rows if row["id"] not in completed_ids]
    interval = 60.0 / args.requests_per_minute
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        if mode == "w":
            stream.write(json.dumps({"record_type": "metadata", "model": args.model, "provider": "cborg", "base_url": args.base_url, "format": "chat", "input": {"prompt_file": str(args.prompts.resolve()), "prompt_field": args.prompt_field, "id_field": args.id_field}}) + "\n")
        for index, row in enumerate(prompt_rows):
            started = time.perf_counter()
            payload = {"model": args.model, "messages": [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": row["prompt"]}], "temperature": args.temperature, "max_tokens": args.max_tokens}
            try:
                for attempt in range(args.retries + 1):
                    try:
                        response = request_completion(args.base_url, api_key, payload)
                        break
                    except (urllib.error.HTTPError, urllib.error.URLError) as error:
                        if attempt == args.retries:
                            raise
                        time.sleep(min(60.0, 2.0 ** (attempt + 1)))
                content = response["choices"][0]["message"].get("content")
                if not isinstance(content, str):
                    raise RuntimeError("CBORG response did not contain text content")
                result = {"record_type": "completion", "id": row["id"], "prompt": row["prompt"], "output": content, "elapsed_seconds": time.perf_counter() - started, "usage": response.get("usage")}
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, KeyError, json.JSONDecodeError) as error:
                result = {"record_type": "completion", "id": row["id"], "prompt": row["prompt"], "output": "", "elapsed_seconds": time.perf_counter() - started, "error": str(error)}
            stream.write(json.dumps(result) + "\n"); stream.flush()
            print(json.dumps({"index": index + 1, "total": len(prompt_rows), "id": row["id"], "elapsed_seconds": round(result["elapsed_seconds"], 2), "error": result.get("error")}))
            if index + 1 < len(prompt_rows):
                time.sleep(max(0.0, interval - result["elapsed_seconds"]))


if __name__ == "__main__":
    main()
