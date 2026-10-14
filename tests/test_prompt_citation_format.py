"""Every instruction an agent reads asks for the citation format the guard
accepts.

Found on the first real-Gemini run of Ryde's DISP-001: the advocate prompts
still said "Cite E# and P-.. IDs", the model wrote "P-§1.2", the guard
rejected it as an unknown citation and the case escalated. Same class as the
earlier P-BF-1 defect: the system asks an agent for something its own guard
will not accept. This test reads every string literal in the agent and LLM
modules and fails on any instruction to cite the retired P- clause IDs.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "ridecourt"
FILES = sorted((ROOT / "agents").glob("*.py")) + [ROOT / "llm.py", ROOT / "orchestrator.py"]
RETIRED = re.compile(r"\bP-(?:\.\.|#|[A-Z]{2}-\d)")


def _string_literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # skip docstrings: they are not shown to a model
            yield node.lineno, node.value


def _docstring_lines(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                lines.add(node.body[0].value.lineno)
    return lines


def test_no_prompt_asks_for_retired_p_ids():
    found = []
    for path in FILES:
        docs = _docstring_lines(path)
        for lineno, text in _string_literals(path):
            if lineno in docs:
                continue
            if RETIRED.search(text):
                found.append(f"{path.name}:{lineno}: {text[:80]!r}")
    assert not found, "Prompts still ask for P- clause IDs:\n" + "\n".join(found)


def test_the_test_catches_the_old_wording(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('PROMPT = "Cite E# and P-.. IDs."\n', encoding="utf-8")
    assert any(RETIRED.search(t) for _, t in _string_literals(bad))
