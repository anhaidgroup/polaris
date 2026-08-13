"""Generate candidate table descriptions with Llama-3.1-8B.

Loads the base model 4-bit quantized, optionally wraps it with a
DPO-trained LoRA adapter, builds the Table 6 prompt for each table, and
samples one or more description candidates per table with the paper's
decoding settings (temperature 0.5, top_p 0.9, top_k 50, typical_p 0.95,
seed 42, max_new_tokens 2048).  Runs on unsloth; the
GPU stack is imported lazily (``pip install -r requirements-gpu.txt``).
Output rows are streamed to disk, and a rerun skips tables already
present in the output file.

Run:  python -m polaris.generate_candidates --datasets arctic ecir lter wikitables wtr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from .prompts.description_generation import (
    TableMeta,
    build_table_info,
    prepare_tbl_desc,
)

_GPU_REMEDY = "pip install -r requirements-gpu.txt"


class DescriptionGenerator:
    """Generates Table 6 JSON descriptions for tables with Llama-3.1-8B."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        adapter: str | None = None,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit: bool = True,
        max_seq_length: int = 8196,  # 8196, not 8192 - do not "fix"
    ) -> "DescriptionGenerator":
        """Load the base model, plus the LoRA adapter when given.

        unsloth may substitute its own pre-quantized 4-bit build of the
        base checkpoint.
        """
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError as exc:
            raise ImportError(
                f"unsloth is required to load the model ({exc}). "
                f"Install it with: {_GPU_REMEDY}"
            ) from exc

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)

        FastLanguageModel.for_inference(model)
        # generate_candidates slices completions at the batch width,
        # which is only correct when prompts are left-padded.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return cls(model, tokenizer)

    def describe(
        self,
        tables: list[TableMeta],
        num_candidates: int = 1,
        batch_size: int = 8,
        temperature: float = 0.5,
        top_p: float = 0.9,
        top_k: int = 50,
        typical_p: float | None = 0.95,
        seed: int | None = 42,
        max_new_tokens: int = 2048,
        expansions: dict[str, dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generate description candidates for each table.

        expansions comes from expand_names (per-table expanded names).
        Yields, per table in input order,
        {"table_id": ..., "candidates": [...], "description": ...} where
        description is the first candidate with its leading ```json
        fence stripped.
        """
        expansions = expansions or {}
        chats = []
        for meta in tables:
            exp = expansions.get(meta.table_id, {})
            chats.append(
                prepare_tbl_desc(
                    build_table_info(
                        meta,
                        column_name_expansion=exp.get("column_name_expansion"),
                        table_name_expansion=exp.get("table_name_expansion"),
                    )
                )
            )
        raw = generate_candidates(
            self.model,
            self.tokenizer,
            chats,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            typical_p=typical_p,
            num_candidates=num_candidates,
            seed=seed,
        )
        for meta, record in zip(tables, raw):
            candidates = record["candidates"]
            yield {
                "table_id": meta.table_id,
                "candidates": candidates,
                "description": candidates[0].split('```json')[-1],
            }


def generate_candidates(
    model: Any,
    tokenizer: Any,
    chat_batches: list[list[dict[str, str]]],
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 50,
    typical_p: float | None = 0.95,
    num_candidates: int = 1,
    seed: int | None = 42,
) -> Iterator[dict[str, Any]]:
    """Sample num_candidates completions per chat, in mini-batches.

    Yields {"messages": chat, "candidates": [str, ...]} per chat, in
    input order, as each mini-batch finishes. The tokenizer must
    left-pad: every prompt then ends at the encoded batch width, and the
    completion is whatever follows it.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"torch is required for description generation ({exc}). "
            f"Install it with: {_GPU_REMEDY}"
        ) from exc

    if seed is not None:
        torch.manual_seed(seed)

    model.eval()

    total_batches = (len(chat_batches) + batch_size - 1) // batch_size
    device = getattr(model, "device", "cuda")

    for start in _progress(
        range(0, len(chat_batches), batch_size),
        total=total_batches,
        desc="generating candidates",
    ):
        chunk = chat_batches[start:start + batch_size]

        rendered = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            for chat in chunk
        ]
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=num_candidates,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                typical_p=typical_p,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # generate() interleaves candidates: [p0_c0, p0_c1, p1_c0, ...]
        batch_rows = enc["input_ids"].size(0)
        prompt_len = enc["input_ids"].shape[1]
        per_prompt: list[list[str]] = [[] for _ in range(batch_rows)]

        for i in range(out.size(0)):
            base_idx = i // num_candidates
            gen_ids = out[i, prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            per_prompt[base_idx].append(text)

        for chat, cands in zip(chunk, per_prompt):
            cands = [c.strip() for c in cands[:num_candidates]]
            yield {
                "messages": chat,
                "candidates": cands,
            }


def _progress(iterable: Iterable, total: int, desc: str) -> Iterable:
    """tqdm progress bar if tqdm is installed, otherwise the iterable unchanged."""
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def run_generation(
    names: list[str],
    num_candidates: int,
    adapter: str | None,
    batch_size: int,
    stem: str,
) -> int:
    """Generate for each dataset, writing {stem}_{name}.jsonl per dataset.

    Streams rows to disk and skips tables already present in the output
    file, so an interrupted run resumes where it stopped.
    """
    from .expand_names import resolve_expansions
    from .load_data import load_dataset_or_exit

    # Plan the work first; the model is loaded only if something is left.
    plans = []
    for name in dict.fromkeys(names):
        tables = load_dataset_or_exit(name).tables
        expansions = resolve_expansions(name)
        out = Path(f"{stem}_{name}.jsonl")
        done: set[str] = set()
        if out.is_file():
            with open(out, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("adapter") != adapter:
                        raise SystemExit(
                            f"{out} was generated with "
                            f"adapter={row.get('adapter')} - delete it "
                            f"before regenerating with a different model"
                        )
                    done.add(str(row["table_id"]))
        todo = [t for t in tables if t.table_id not in done]
        if done:
            print(f"{out}: {len(done)} rows already present, {len(todo)} to go")
        if todo:
            plans.append((todo, expansions, out, len(done)))
    if not plans:
        print("Nothing to do")
        return 0

    generator = DescriptionGenerator.from_pretrained(adapter=adapter)
    for todo, expansions, out, n in plans:
        with open(out, "a", encoding="utf-8") as f:
            try:
                for row in generator.describe(
                    todo,
                    num_candidates=num_candidates,
                    batch_size=batch_size,
                    expansions=expansions,
                ):
                    row["adapter"] = adapter
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
            except Exception as exc:
                # torch.cuda.OutOfMemoryError, matched by name
                if type(exc).__name__ == "OutOfMemoryError":
                    raise SystemExit(
                        "GPU out of memory during generation - lower "
                        f"--batch-size (currently {batch_size}); the "
                        f"{n} rows already in {out} are kept, rerunning "
                        "resumes after them"
                    ) from None
                raise
        print(f"{out}: {n} rows")
    return 0


def main(argv=None) -> int:
    from .utils.env import load_env

    load_env()
    parser = argparse.ArgumentParser(
        prog="python -m polaris.generate_candidates",
        description="Generate candidate table descriptions for training "
        "(GPU; pip install -r requirements-gpu.txt).",
    )
    parser.add_argument(
        "--datasets", metavar="NAME", nargs="+", required=True,
        help="datasets under data/; writes candidates_{NAME}.jsonl each",
    )
    parser.add_argument(
        "--candidates", type=int, default=3,
        help="candidates per table (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="tables per GPU batch (lower it on GPU out-of-memory)",
    )
    args = parser.parse_args(argv)
    return run_generation(
        args.datasets, args.candidates, None, args.batch_size, "candidates"
    )


if __name__ == "__main__":
    sys.exit(main())
