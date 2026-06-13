#!/usr/bin/env python3
"""Build the source-clarity closure board and apply safe wording cleanups."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "docs" / "source-clarity-closure-board.md"

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".rs",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".xlog",
}

DOC_EXTENSIONS = {".md", ".rst", ".txt", ".tex"}

SOURCE_FILENAMES = {
    "Cargo.toml",
    "Makefile",
    "justfile",
    "pyproject.toml",
    "requirements.txt",
}

EXCLUDED_PARTS = {
    ".git",
    ".worktrees",
    "target",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "superpowers",
    "plans",
    "evidence",
    "artifacts",
    "results",
    "outputs",
}

COMMON_UPPERCASE = {
    "API",
    "CLI",
    "CPU",
    "CSV",
    "CUDA",
    "GPU",
    "HTTP",
    "JSON",
    "PDF",
    "README",
    "SQL",
    "TOML",
    "URI",
    "URL",
    "UTF",
    "UUID",
    "XML",
    "XLOG",
    "YAML",
}

PROJECT_TERMS = {
    "BFO": "external biomedical-ontology diagnostic package",
    "CSM": "counting sort merge",
    "D2H": "device-to-host transfer",
    "DILP": "differentiable inductive logic programming",
    "DLPack": "tensor interchange protocol for zero-copy array sharing",
    "DNNF": "decomposable negation normal form",
    "DTS": "external consumer",
    "DTS-DLM": "external consumer",
    "EDB": "extensional database relation",
    "EGB": "epistemic gate bundle",
    "EIR": "epistemic intermediate representation",
    "FAEEL": "founded auto-epistemic logic",
    "HG": "hypergraph",
    "IDB": "intensional database relation",
    "ILP": "inductive logic programming",
    "LWM": "living-world model diagnostic package",
    "MC": "Monte Carlo",
    "NAF": "negation as failure",
    "PIR": "probabilistic intermediate representation",
    "PTX": "NVIDIA CUDA virtual instruction assembly",
    "RIR": "relational intermediate representation",
    "RunPod": "remote GPU execution provider",
    "SCC": "strongly connected component",
    "UCR": "external case-reasoner diagnostic package",
    "WCOJ": "worst-case optimal join",
    "WFS": "well-founded semantics",
    "WMC": "weighted model counting",
    "cuDF": "GPU dataframe library",
}

TECHNICAL_TERM_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(sorted((re.escape(k) for k in PROJECT_TERMS), key=len, reverse=True))
    + r"|dILP|D2H|H2D|PTX|DLPack|cuDF|RunPod)\b"
)

TASK_CODE_PATTERN = re.compile(
    r"\b(?:"
    r"v\d{3}|v\d+\.\d+(?:\.\d+)?|"
    r"(?:FRS|REQ|EF|G|W|M|P|D|S|L|B|C)\d{1,3}[A-Za-z]?"
    r"(?:[.+-][A-Z]?\d+[A-Za-z]?|[.+-][A-Z])*"
    r")\b"
)

CONSUMER_REPLACEMENTS = (
    (re.compile(r"\bDTS-DLM\b"), "external consumer"),
    (re.compile(r"\bDTS\b"), "external consumer"),
)


@dataclass
class FileScan:
    path: str
    comment_count: int
    code_count: int
    comment_terms: Counter[str]
    code_terms: Counter[str]

    @property
    def resolved(self) -> bool:
        return self.comment_count == 0 and self.code_count == 0


def git_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line]


def is_eligible(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDED_PARTS:
        return False
    if BOARD_PATH.exists() and path == BOARD_PATH:
        return False
    if rel.name in SOURCE_FILENAMES:
        return True
    return path.suffix in SOURCE_EXTENSIONS or path.suffix in DOC_EXTENSIONS


def split_markdown(text: str) -> tuple[str, str]:
    prose: list[str] = []
    code: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        (code if in_fence else prose).append(line)
    return "\n".join(prose), "\n".join(code)


def split_code(text: str, suffix: str) -> tuple[str, str]:
    block_comments = "\n".join(re.findall(r"/\*.*?\*/", text, flags=re.S))
    without_blocks = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    comments: list[str] = [block_comments]
    code: list[str] = []
    for line in without_blocks.splitlines():
        stripped = line.lstrip()
        marker = None
        if suffix in {".rs", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
            marker = "//"
        elif suffix in {".py", ".sh", ".toml", ".yaml", ".yml"}:
            marker = "#"
        if marker and marker in line:
            before, after = line.split(marker, 1)
            code.append(before)
            comments.append(after)
        else:
            code.append(line)
    return "\n".join(comments), "\n".join(code)


def normalize_term(term: str) -> str:
    if term == "dILP":
        return "DILP"
    return term


def scan_terms(text: str) -> Counter[str]:
    terms: Counter[str] = Counter()
    for match in TECHNICAL_TERM_PATTERN.finditer(text):
        value = normalize_term(match.group(0))
        if value not in COMMON_UPPERCASE:
            terms[value] += 1
    for match in TASK_CODE_PATTERN.finditer(text):
        value = match.group(0)
        if value not in COMMON_UPPERCASE and not value.startswith("UTF"):
            terms[value] += 1
    return terms


def scan_file(path: Path) -> FileScan:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix in DOC_EXTENSIONS:
        comment_text, code_text = split_markdown(text)
    else:
        comment_text, code_text = split_code(text, path.suffix)
    comment_terms = scan_terms(comment_text)
    code_terms = scan_terms(code_text)
    return FileScan(
        path=rel,
        comment_count=sum(comment_terms.values()),
        code_count=sum(code_terms.values()),
        comment_terms=comment_terms,
        code_terms=code_terms,
    )


def representative(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return ", ".join(f"{term}({count})" for term, count in counter.most_common(6))


def render_board(scans: list[FileScan]) -> str:
    total_comment = sum(scan.comment_count for scan in scans)
    total_code = sum(scan.code_count for scan in scans)
    unresolved = sum(1 for scan in scans if not scan.resolved)
    lines = [
        "# Source Clarity Closure Board",
        "",
        "Scope: Git-tracked source and documentation files only. Superpowers docs, plan files, evidence directories, generated artifacts, result/output directories, build output, and other worktrees are excluded.",
        "",
        f"Scanned files: {len(scans)}",
        f"Unresolved files: {unresolved}",
        f"Comment/prose artifact occurrences: {total_comment}",
        f"Code/identifier artifact occurrences: {total_code}",
        "",
        "Resolved means this scan found no remaining opaque project artifact codes, consumer names, or project-specific abbreviations in the eligible portions of that file.",
        "",
        "## Term Meanings Used For Resolution",
        "",
        "| Artifact | Meaning used during cleanup |",
        "|---|---|",
    ]
    for term, meaning in sorted(PROJECT_TERMS.items()):
        lines.append(f"| `{term}` | {meaning} |")
    lines.extend(
        [
            "",
            "## File Board",
            "",
            "| File path | Artifacts found in comments/prose count | Artifacts found in code/naming count | Resolved | Representative comment/prose artifacts | Representative code/naming artifacts |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for scan in sorted(scans, key=lambda item: item.path):
        lines.append(
            f"| `{scan.path}` | {scan.comment_count} | {scan.code_count} | "
            f"{str(scan.resolved).lower()} | {representative(scan.comment_terms)} | {representative(scan.code_terms)} |"
        )
    lines.append("")
    return "\n".join(lines)


def apply_consumer_replacements(paths: list[Path]) -> int:
    changed = 0
    for path in paths:
        rel = path.relative_to(ROOT)
        if not is_eligible(path):
            continue
        if rel.as_posix() == "tools/source_clarity_board.py":
            continue
        if path.suffix not in DOC_EXTENSIONS and path.suffix not in SOURCE_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        new_text = text
        for pattern, replacement in CONSUMER_REPLACEMENTS:
            new_text = pattern.sub(replacement, new_text)
        new_text = re.sub(r"\ba external consumer\b", "an external consumer", new_text)
        new_text = re.sub(r"\bA external consumer\b", "An external consumer", new_text)
        new_text = new_text.replace("external-consumer-first", "external-consumer-first")
        new_text = new_text.replace("external-consumer-focused", "external-consumer-focused")
        new_text = new_text.replace("External Consumer Release Gates", "External Consumer Release Gates")
        new_text = new_text.replace("external consumer sends", "an external consumer sends")
        new_text = new_text.replace("external consumer-Fit", "External Consumer Fit")
        new_text = new_text.replace("external consumer-FIT", "EXTERNAL CONSUMER FIT")
        new_text = new_text.replace("same as external consumer ", "same as the external consumer ")
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
            print(rel.as_posix())
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-consumer-names", action="store_true")
    args = parser.parse_args()
    files = [path for path in git_files() if is_eligible(path)]
    if args.fix_consumer_names:
        changed = apply_consumer_replacements(files)
        print(f"consumer-name files changed: {changed}")
    scans = [scan_file(path) for path in files]
    BOARD_PATH.write_text(render_board(scans), encoding="utf-8")
    print(f"wrote {BOARD_PATH.relative_to(ROOT)} with {len(scans)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
