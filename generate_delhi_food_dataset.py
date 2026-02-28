from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import RateLimitError, APIConnectionError, ServiceUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
CHUNKS_PATH_DEFAULT = PROJECT_ROOT / "chunks.json"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "dataset" / "_processed"

KEY_NAMES = [f"GEMINI_API_KEY{i}" for i in range(1, 11)]

QUESTION_PROMPT = (
    "Generate one conversation initiating statement in English/Hinglish based on this knowledge. "
    "Use varied starts like why/when/where/how, and keep it natural and specific.\n\n"
    "KNOWLEDGE:\n{knowledge}"
)

ANSWER_PROMPT = (
    "Using the knowledge, answer in 2 concise informative lines in English. "
    "Then ask one thoughtful follow-up question in English/Hinglish. "
    "Do not generate extra turns.\n\n"
    "KNOWLEDGE:\n{knowledge}\n\n"
    "USER QUESTION:\n{question}"
)


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("chunks.json must be a list of chunk objects")
    return data


def load_done_pairs(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                chunk_index = int(row["chunk_index"])
                pair_index = int(row.get("pair_index", 0))
                done.add((chunk_index, pair_index))
            except Exception:
                continue
    return done


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def trim_knowledge(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


@dataclass
class KeyState:
    key: str
    min_interval_sec: float
    next_ready: float = 0.0
    fail_count: int = 0

    async def wait_ready(self):
        now = time.time()
        if self.next_ready > now:
            await asyncio.sleep(self.next_ready - now)

    def mark_success(self):
        self.fail_count = 0
        self.next_ready = time.time() + self.min_interval_sec

    def mark_rate_limited(self, cooldown_sec: int):
        self.fail_count += 1
        self.next_ready = time.time() + cooldown_sec

    def mark_transient_error(self, penalty_sec: int = 4):
        self.fail_count += 1
        self.next_ready = time.time() + penalty_sec


async def call_model(
    prompt: str,
    state: KeyState,
    model: str,
    temperature: float,
    timeout: int,
    max_attempts: int,
    cooldown_sec: int,
) -> str:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        await state.wait_ready()
        try:
            resp = await asyncio.to_thread(
                completion,
                model=model,
                api_key=state.key,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=timeout,
            )
            text = resp.choices[0].message.content.strip()
            state.mark_success()
            return text
        except RateLimitError as e:
            last_error = e
            state.mark_rate_limited(cooldown_sec)
        except (APIConnectionError, ServiceUnavailableError) as e:
            last_error = e
            state.mark_transient_error()
        except Exception as e:
            last_error = e
            state.mark_transient_error()

    raise RuntimeError(f"All retries failed. Last error: {last_error}")


async def worker(
    worker_id: int,
    state: KeyState,
    queue: asyncio.Queue,
    write_lock: asyncio.Lock,
    progress_lock: asyncio.Lock,
    args: argparse.Namespace,
    base_done_count: int,
    progress_state: dict,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        chunk_index, pair_index, row = item
        knowledge_raw = (row.get("content") or "").strip()
        knowledge = trim_knowledge(knowledge_raw, args.max_knowledge_chars)

        try:
            q_prompt = QUESTION_PROMPT.format(knowledge=knowledge)
            question = await call_model(
                q_prompt,
                state,
                model=args.model,
                temperature=args.q_temperature,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                cooldown_sec=args.cooldown_sec,
            )

            a_prompt = ANSWER_PROMPT.format(knowledge=knowledge, question=question)
            answer = await call_model(
                a_prompt,
                state,
                model=args.model,
                temperature=args.a_temperature,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                cooldown_sec=args.cooldown_sec,
            )

            out = {
                "chunk_index": chunk_index,
                "pair_index": pair_index,
                "source_metadata": row.get("metadata", {}),
                "question": question,
                "answer": answer,
                "question_key": state.key[-6:],
                "answer_key": state.key[-6:],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
            }

            async with write_lock:
                append_jsonl(args.checkpoint_jsonl, out)

            async with progress_lock:
                progress_state["created"] += 1
                created = progress_state["created"]
                if created % args.progress_every == 0:
                    write_progress(
                        args.progress_json,
                        {
                            "started_at": progress_state["started_at"],
                            "last_update": datetime.now(timezone.utc).isoformat(),
                            "created_this_run": created,
                            "total_done": base_done_count + created,
                            "checkpoint": str(args.checkpoint_jsonl),
                        },
                    )

        except Exception as e:
            async with write_lock:
                append_jsonl(
                    args.errors_jsonl,
                    {
                        "chunk_index": chunk_index,
                        "pair_index": pair_index,
                        "error": str(e),
                        "at": datetime.now(timezone.utc).isoformat(),
                        "worker": worker_id,
                    },
                )

        queue.task_done()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Delhi-food synthetic Q/A pairs quickly.")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash-lite")
    parser.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None)
    parser.add_argument("--errors-jsonl", type=Path, default=None)
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pairs-per-chunk", type=int, default=1)
    parser.add_argument("--min-text-chars", type=int, default=120)
    parser.add_argument("--max-knowledge-chars", type=int, default=3500)
    parser.add_argument("--min-interval", type=float, default=3.5)
    parser.add_argument("--cooldown-sec", type=int, default=35)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--q-temperature", type=float, default=0.7)
    parser.add_argument("--a-temperature", type=float, default=0.4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(ENV_PATH)
    keys = [os.getenv(k) for k in KEY_NAMES if os.getenv(k)]
    if not keys:
        raise ValueError("No GEMINI_API_KEY1..10 keys found in .env")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_jsonl = args.checkpoint_jsonl or (args.out_dir / "qa_pairs_checkpoint.jsonl")
    args.errors_jsonl = args.errors_jsonl or (args.out_dir / "qa_pairs_errors.jsonl")
    args.progress_json = args.progress_json or (args.out_dir / "qa_pairs_progress.json")

    chunks = load_chunks(args.chunks_path)
    done_pairs = load_done_pairs(args.checkpoint_jsonl)
    base_done_count = len(done_pairs)

    if args.shuffle:
        import random

        random.shuffle(chunks)

    queue: asyncio.Queue = asyncio.Queue()
    scheduled = 0
    for idx, row in enumerate(chunks):
        knowledge = (row.get("content") or "").strip()
        if len(knowledge) < args.min_text_chars:
            continue
        for pair_index in range(args.pairs_per_chunk):
            if (idx, pair_index) in done_pairs:
                continue
            await queue.put((idx, pair_index, row))
            scheduled += 1
            if args.max_pairs is not None and scheduled >= args.max_pairs:
                break
        if args.max_pairs is not None and scheduled >= args.max_pairs:
            break

    if scheduled == 0:
        print("Nothing to do. All requested pairs already exist.")
        return 0

    print("Model:", args.model)
    print("Keys found:", len(keys))
    print("Chunks loaded:", len(chunks))
    print("Already done:", base_done_count)
    print("Scheduled this run:", scheduled)

    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    progress_state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "created": 0,
    }

    workers = [
        asyncio.create_task(
            worker(
                i,
                KeyState(key=k, min_interval_sec=args.min_interval),
                queue=queue,
                write_lock=write_lock,
                progress_lock=progress_lock,
                args=args,
                base_done_count=base_done_count,
                progress_state=progress_state,
            )
        )
        for i, k in enumerate(keys)
    ]

    for _ in workers:
        await queue.put(None)

    await queue.join()
    await asyncio.gather(*workers)

    created = progress_state["created"]
    write_progress(
        args.progress_json,
        {
            "started_at": progress_state["started_at"],
            "last_update": datetime.now(timezone.utc).isoformat(),
            "created_this_run": created,
            "total_done": base_done_count + created,
            "checkpoint": str(args.checkpoint_jsonl),
            "errors_file": str(args.errors_jsonl),
        },
    )

    print("Run complete")
    print("Created this run:", created)
    print("Total checkpointed:", base_done_count + created)
    print("Checkpoint file:", args.checkpoint_jsonl)
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
