"""Fine-tune the description model with DPO on the preference pairs.

Consumes the pairs jsonl written by ``polaris.generate_preference_pairs``
(one ``{"prompt": [chat messages], "chosen": str, "rejected": str}`` per
line). Training loads the base model 4-bit quantized via unsloth,
attaches a LoRA adapter, and runs ``trl.DPOTrainer`` with the paper's
Table 8 hyperparameters (see :data:`DPO_DEFAULTS`).

Run:  python -m polaris.dpo   # reads pairs.jsonl, saves the adapter to out/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .utils import line_buffered

#: Training settings (paper Table 8).
DPO_DEFAULTS: dict[str, Any] = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "output_dir": "out/adapter",  # trained LoRA adapter + tokenizer land here
    "dataset": "pairs.jsonl",  # {"prompt": [chat messages], "chosen": str, "rejected": str} per line
    "load_in_4bit": True,
    "max_seq_length": 8196,  # 8196, not 8192 - do not "fix"
    # LoRA
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # DPO
    "beta": 0.1,
    "learning_rate": 5.0e-6,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,  # effective batch 8 = 4 x grad-accum 2
    "gradient_accumulation_steps": 2,
    "optim": "adamw_8bit",
    "weight_decay": 0.0,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "logging_steps": 1,
    "seed": 42,
    "lora_random_state": 3407,  # LoRA init seed
    # prompts (~1,200+ tokens) are truncated by TRL to these limits
    "max_length": 1024,
    "max_prompt_length": 512,
}

#: Some pairs files prefix completions with this; stripped before training.
_ASSISTANT_PREFIX = "<|assistant|>\n"


def train(overrides: dict[str, Any] | None = None) -> str:
    """Run DPO training.

    Args:
        overrides: flat ``{key: value}`` overrides applied on top of
            :data:`DPO_DEFAULTS`; unknown keys are rejected to catch typos.

    Returns:
        The output directory containing the trained adapter and tokenizer.
    """
    try:
        # unsloth must be imported before torch/trl for its patches to apply
        from unsloth import FastLanguageModel, PatchDPOTrainer
        from unsloth.chat_templates import get_chat_template
        import torch
        import trl
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(
            f"DPO training needs the GPU stack ({exc}). "
            f"Install it with: pip install -r requirements-gpu.txt"
        ) from exc

    cfg = dict(DPO_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise KeyError(
                f"Unknown config override(s) {sorted(unknown)}; "
                f"valid keys: {sorted(cfg)}"
            )
        cfg.update(overrides)

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    PatchDPOTrainer()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=int(cfg["max_seq_length"]),
        dtype=None,
        load_in_4bit=cfg.get("load_in_4bit", True),
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        target_modules=list(cfg["lora_target_modules"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing=True,
        random_state=int(cfg["lora_random_state"]),
    )

    pairs = load_pairs(cfg["dataset"])
    print(f"Training on {len(pairs)} preference pairs")
    records = [format_pair(p, tokenizer) for p in pairs]
    train_dataset = hf_datasets.Dataset.from_list(records)

    dpo_args = trl.DPOConfig(
        save_strategy="no",  # only the final adapter is saved, below
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        fp16=not bf16,
        bf16=bf16,
        logging_steps=int(cfg.get("logging_steps", 1)),
        optim=cfg["optim"],
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=int(cfg["seed"]),
        output_dir=cfg["output_dir"],
        beta=float(cfg["beta"]),
        max_length=int(cfg.get("max_length", 1024)),
        max_prompt_length=int(cfg.get("max_prompt_length", 512)),
        padding_value=tokenizer.pad_token_id,
    )

    # trl>=0.12 takes the tokenizer as `processing_class`.
    trainer = trl.DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        effective = int(cfg["per_device_train_batch_size"]) * int(
            cfg["gradient_accumulation_steps"]
        )
        raise SystemExit(
            "GPU out of memory during DPO training - lower --batch-size "
            f"(currently {cfg['per_device_train_batch_size']}) and raise "
            f"--grad-accum to keep the effective batch at {effective}"
        ) from None

    output_dir = cfg["output_dir"]
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return str(output_dir)


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Read a preference-pairs jsonl file.

    Each line must hold ``{"prompt": [chat messages], "chosen": str,
    "rejected": str}``; ``prompts`` is accepted as an alias for
    ``prompt``. Returns the pairs as a list of dicts.
    """
    pairs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", row.get("prompts"))
            if prompt is None or "chosen" not in row or "rejected" not in row:
                raise ValueError(
                    f"{path}:{line_no}: expected keys prompt/chosen/rejected"
                )
            pairs.append(
                {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}
            )
    if not pairs:
        raise ValueError(f"{path}: no preference pairs found")
    return pairs


def format_pair(
    example: dict[str, Any], tokenizer: Any, assistant_prefix: str = _ASSISTANT_PREFIX
) -> dict[str, str]:
    """Render one preference pair to the three strings DPOTrainer expects.

    The prompt messages are cut off after the last user turn, an empty
    system message is prepended when one is missing, and the result is
    rendered with the tokenizer's chat template plus a generation prompt.
    The chosen/rejected completions have any leading ``"<|assistant|>\\n"``
    stripped. Returns ``{"prompt": str, "chosen": str, "rejected": str}``.
    """
    prompt_msgs = _messages_up_to_last_user(list(example["prompt"]))
    if not prompt_msgs or prompt_msgs[0]["role"] != "system":
        prompt_msgs = [{"role": "system", "content": ""}] + prompt_msgs

    text_prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    def _strip_prefix(s: str) -> str:
        return s.removeprefix(assistant_prefix)

    return {
        "prompt": text_prompt,
        "chosen": _strip_prefix(example["chosen"]),
        "rejected": _strip_prefix(example["rejected"]),
    }


def _messages_up_to_last_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the chat history up to and including the last user turn.

    The history must end on that user turn — the completion strings stand
    in for the assistant reply that would follow it.
    """
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise ValueError("No user message found in chat.")
    if last_user_idx != len(messages) - 1:
        raise ValueError("prompt must end with a user message; found messages after the last user turn")
    return messages[: last_user_idx + 1]


def main(argv=None) -> int:
    from .utils.env import load_env

    load_env()
    line_buffered()
    parser = argparse.ArgumentParser(
        prog="python -m polaris.dpo",
        description="DPO-finetune the description model on preference pairs "
        "(paper Table 8 settings; GPU required).",
    )
    parser.add_argument(
        "--data", default=DPO_DEFAULTS["dataset"],
        help="training pairs jsonl from generate_preference_pairs "
        "(default: pairs.jsonl)",
    )
    parser.add_argument(
        "--output-dir", default=DPO_DEFAULTS["output_dir"],
        help="where the adapter is saved (default: out/adapter)",
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=DPO_DEFAULTS["per_device_train_batch_size"],
        help="per-device train batch size (lower it on GPU out-of-memory)",
    )
    parser.add_argument(
        "--grad-accum", type=int,
        default=DPO_DEFAULTS["gradient_accumulation_steps"],
        help="gradient accumulation steps (raise it when lowering --batch-size)",
    )
    args = parser.parse_args(argv)
    if not Path(args.data).is_file():
        raise SystemExit(
            f"{args.data} not found - generate it first: python -m "
            f"polaris.generate_preference_pairs --datasets <training datasets>"
        )
    try:
        output_dir = train({
            "dataset": args.data,
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
        })
    except ImportError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Adapter saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
