"""Run the agent against eval/qa_dataset.jsonl and dump results to a CSV.

P1: walks dataset, invokes the (stubbed) graph, writes results. P6 adds
baseline runners (text RAG, ColPali + single-shot LLM) and metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.graph import compile_graph


def run(dataset: Path, out_csv: Path) -> None:
    app = compile_graph()
    rows = []
    with dataset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa = json.loads(line)
            init = {
                "query": qa["question"],
                "plan_cursor": 0,
                "reflexion_iter": 0,
                "is_sufficient": False,
                "retrieved_pages": [],
                "extracted_facts": [],
            }
            out = app.invoke(init)
            rows.append({
                "id": qa["id"],
                "level": qa["level"],
                "question": qa["question"],
                "gold": qa["answer"],
                "pred": out.get("answer", ""),
            })
            print(f"{qa['id']}: ok")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "qa_dataset.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.csv")
    args = parser.parse_args()
    run(args.dataset, args.out)


if __name__ == "__main__":
    main()
