from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_source(meta: dict) -> str:
    title = (meta or {}).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    src = (meta or {}).get("source")
    if isinstance(src, str) and src.strip():
        return Path(src).name
    return "delhi-food"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export qa_pairs_checkpoint.jsonl to FineTome-style JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dataset/_processed/qa_pairs_checkpoint.jsonl"),
        help="Path to qa_pairs_checkpoint.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset/delhi_food_finetome_train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=5.0,
        help="Score value to attach to each row (float).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            out = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
                "source": resolve_source(row.get("source_metadata") or {}),
                "score": float(args.score),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print("Input rows:", total)
    print("Written rows:", written)
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
