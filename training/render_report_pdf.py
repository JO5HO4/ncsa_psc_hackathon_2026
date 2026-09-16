#!/usr/bin/env python3
"""Render a campaign Markdown report and its local PNG figures into one PDF."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap


PAGE = (8.5, 11)
LEFT, RIGHT, TOP, BOTTOM = 0.72, 0.68, 0.70, 0.62


def plain(value: str) -> str:
    value = re.sub(r"!\[[^]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    return value.replace("**", "").replace("`", "")


class Document:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.page_number = 0
        self.fig = None
        self.y = 0
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.33, str(self.page_number), ha="center", va="center", fontsize=8, color="#555555")
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_number += 1
        self.fig = plt.figure(figsize=PAGE)
        self.fig.patch.set_facecolor("white")
        self.y = 1 - TOP / PAGE[1]

    def space_for(self, height: float):
        if self.y - height < BOTTOM / PAGE[1]:
            self.new_page()

    def text(self, value: str, *, size=10.2, weight="normal", mono=False, gap=0.010):
        chars = 92 if not mono else 101
        lines = wrap(value, width=chars, break_long_words=False, break_on_hyphens=False) or [""]
        line_height = (size / 72) / PAGE[1] * 1.40
        height = len(lines) * line_height + gap
        self.space_for(height)
        self.fig.text(LEFT / PAGE[0], self.y, "\n".join(lines), va="top", ha="left", fontsize=size,
                      fontweight=weight, family="monospace" if mono else "sans-serif")
        self.y -= height

    def heading(self, value: str, level: int):
        size = {1: 18, 2: 14, 3: 11.5}[min(level, 3)]
        self.space_for((size / 72) / PAGE[1] * 2.2)
        self.text(value, size=size, weight="bold", gap=0.014)

    def image(self, path: Path, label: str):
        if self.y < 0.72:
            self.new_page()
        image = mpimg.imread(path)
        ratio = image.shape[0] / image.shape[1]
        width = 1 - (LEFT + RIGHT) / PAGE[0]
        height = min(width * ratio, 0.64)
        self.fig.text(LEFT / PAGE[0], self.y, label, va="top", ha="left", fontsize=11, fontweight="bold")
        self.y -= 0.035
        ax = self.fig.add_axes([LEFT / PAGE[0], self.y - height, width, height])
        ax.imshow(image)
        ax.axis("off")
        self.y -= height + 0.025


def render(source: Path, output: Path):
    lines = source.read_text().splitlines()
    with PdfPages(output, metadata={"Title": "Qwen3.5-9B ROOT-command LoRA results", "Author": "NCSA/PSC hackathon"}) as pdf:
        doc = Document(pdf)
        table: list[str] = []

        def flush_table():
            nonlocal table
            if not table:
                return
            rows = []
            for row in table:
                cells = [plain(c.strip()) for c in row.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                rows.append(" | ".join(cells))
            for row in rows:
                doc.text(row, size=8.0, mono=True, gap=0.002)
            doc.text("", size=3, gap=0.006)
            table = []

        for raw in lines:
            if raw.startswith("|"):
                table.append(raw)
                continue
            flush_table()
            match = re.match(r"^(#{1,3})\s+(.*)$", raw)
            if match:
                doc.heading(plain(match.group(2)), len(match.group(1)))
                continue
            image = re.match(r"^!\[([^]]*)\]\(([^)]*)\)$", raw)
            if image:
                path = (source.parent / image.group(2)).resolve()
                if path.is_file():
                    doc.image(path, image.group(1) or path.stem)
                continue
            if raw.startswith("- "):
                doc.text("• " + plain(raw[2:]), gap=0.004)
            elif raw.strip():
                doc.text(plain(raw))
            else:
                doc.y -= 0.006
        flush_table()
        doc.new_page()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} REPORT.md OUTPUT.pdf")
    source, output = map(Path, sys.argv[1:])
    if not source.is_file():
        raise SystemExit(f"Report not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(source, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
