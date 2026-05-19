from dataclasses import dataclass,field
from pathlib import Path
import json

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)

def save_eval_results(results: list[dict], path: str = "eval_results.json") -> None:
    Path(path).write_text(json.dumps(results, indent=2, ensure_ascii=False))

def load_eval_results(path: str = "eval_results.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())