"""
Reward-weighted policy optimization for the WoG agent.

Uses trajectory data with pre-computed reward signals to do
reward-weighted SFT — each example's loss is scaled by its
normalized reward, pushing the model toward high-reward actions
and away from low/negative-reward ones.

This is equivalent to a simplified policy gradient (REINFORCE-style)
without needing an online rollout loop.

W&B logging:
  policy/reward_mean, policy/reward_std, policy/reward_min, policy/reward_max
  policy/positive_examples, policy/negative_examples
  train/loss, train/weighted_loss, train/reward_weight
  eval/loss, eval/perplexity
  charts: reward histogram, loss vs reward scatter

Usage:
    python train_policy_nvidia.py --data data --epochs 3
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_trajectories(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def format_chatml(messages: list[dict]) -> str:
    """Format messages as ChatML prompt."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_training_text(record: dict) -> str:
    """Build full training text: prompt + response."""
    messages = record.get("messages", [])
    response = record.get("response", "")
    prompt = format_chatml(messages)
    return prompt + response + "<|im_end|>"


# ── Reward normalization ──────────────────────────────────────────────────────

def normalize_rewards(rewards: list[float]) -> list[float]:
    """
    Normalize rewards to [0, 2] range centered at 1.0.
    Positive rewards > 1.0 (upweight), negative < 1.0 (downweight).
    Zero reward = 1.0 (neutral).
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    pos = rewards_t[rewards_t > 0]
    neg = rewards_t[rewards_t < 0]

    max_pos = pos.max().item() if len(pos) > 0 else 1.0
    min_neg = neg.min().item() if len(neg) > 0 else -1.0

    normalized = []
    for r in rewards:
        if r > 0:
            normalized.append(1.0 + (r / max_pos))       # [1.0, 2.0]
        elif r < 0:
            normalized.append(1.0 + (r / abs(min_neg)))   # [0.0, 1.0)
        else:
            normalized.append(1.0)                         # neutral
    return normalized


# ── Custom reward-weighted Trainer ───────────────────────────────────────────

class RewardWeightedTrainer(Trainer):
    """
    Trainer subclass that weights each example's loss by its reward.
    High-reward examples get loss_weight > 1 (model learns more from them).
    Negative-reward examples get loss_weight < 1 (model unlearns them).
    """

    def __init__(self, reward_weights: list[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        self._step_reward_weights: list[float] = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract reward weights for this batch
        sample_idx = inputs.pop("sample_idx", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        # Compute per-token cross entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_loss = token_loss.view(shift_labels.size())

        # Mask padding tokens
        mask = (shift_labels != -100).float()
        per_example_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Apply reward weights
        if sample_idx is not None:
            weights = self._reward_weights[sample_idx.cpu()].to(per_example_loss.device)
            weighted_loss = (per_example_loss * weights).mean()
            self._step_reward_weights.extend(weights.cpu().tolist())
        else:
            weighted_loss = per_example_loss.mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def log(self, logs: dict, start_time=None) -> None:
        if self._step_reward_weights:
            avg_w = sum(self._step_reward_weights) / len(self._step_reward_weights)
            logs["train/avg_reward_weight"] = avg_w
            self._step_reward_weights = []
        super().log(logs)


# ── Tokenization ─────────────────────────────────────────────────────────────

def tokenize(examples: list[dict], reward_weights: list[float], tokenizer, max_length: int = 2048) -> Dataset:
    texts = [build_training_text(r) for r in examples]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].copy(),
        "sample_idx": list(range(len(examples))),
        "reward_weight": reward_weights,
    }
    return Dataset.from_dict(data)


# ── W&B reward analysis ───────────────────────────────────────────────────────

def log_reward_analysis(rewards: list[float], split: str = "train") -> None:
    """Log reward distribution charts to W&B."""
    pos = [r for r in rewards if r > 0]
    neg = [r for r in rewards if r < 0]
    zero = [r for r in rewards if r == 0]

    wandb.log({
        f"{split}/reward_mean": sum(rewards) / len(rewards),
        f"{split}/reward_std": torch.tensor(rewards).std().item(),
        f"{split}/reward_min": min(rewards),
        f"{split}/reward_max": max(rewards),
        f"{split}/positive_examples": len(pos),
        f"{split}/negative_examples": len(neg),
        f"{split}/zero_examples": len(zero),
        f"{split}/pct_positive": len(pos) / len(rewards) * 100,
    })

    # Histogram
    table = wandb.Table(columns=["reward"])
    for r in rewards:
        table.add_data(r)
    wandb.log({f"{split}/reward_histogram": wandb.plot.histogram(
        table, "reward", title=f"Reward Distribution ({split})"
    )})

    # Top 10 reward signals breakdown
    signal_means = {}
    signal_keys = ["gold_delta", "xp_delta", "quests_completed_delta",
                   "deaths_delta", "zones_discovered_delta", "quest_gold_delta"]
    # We don't have access to full records here — log summary instead
    wandb.log({f"{split}/reward_analysis": wandb.Table(
        columns=["bucket", "count", "avg_reward"],
        data=[
            ["positive (reward > 0)", len(pos), sum(pos)/len(pos) if pos else 0],
            ["zero (reward == 0)", len(zero), 0],
            ["negative (reward < 0)", len(neg), sum(neg)/len(neg) if neg else 0],
        ]
    )})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output-dir", default="adapters/policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--min-reward-pct", type=float, default=0.0,
                        help="Filter out bottom N%% of reward examples")
    args = parser.parse_args()

    train_path = os.path.join(args.data, "train.jsonl")
    valid_path = os.path.join(args.data, "valid.jsonl")
    for p in [train_path, valid_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            raise SystemExit(1)

    # Load data
    print("Loading trajectories...")
    train_records = load_trajectories(train_path)
    valid_records = load_trajectories(valid_path)
    print(f"  Train: {len(train_records)}, Valid: {len(valid_records)}")

    train_rewards = [r.get("reward", 0.0) for r in train_records]
    valid_rewards = [r.get("reward", 0.0) for r in valid_records]

    # Optional: filter bottom reward percentile
    if args.min_reward_pct > 0:
        threshold = sorted(train_rewards)[int(len(train_rewards) * args.min_reward_pct / 100)]
        before = len(train_records)
        train_records = [r for r in train_records if r.get("reward", 0) >= threshold]
        train_rewards = [r.get("reward", 0.0) for r in train_records]
        print(f"  Filtered {before - len(train_records)} low-reward examples (threshold={threshold:.2f})")

    # Normalize rewards to loss weights
    train_weights = normalize_rewards(train_rewards)
    valid_weights = normalize_rewards(valid_rewards)

    config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_examples": len(train_records),
        "valid_examples": len(valid_records),
        "reward_mean": sum(train_rewards) / len(train_rewards),
        "reward_max": max(train_rewards),
        "reward_min": min(train_rewards),
        "pct_positive_reward": sum(1 for r in train_rewards if r > 0) / len(train_rewards) * 100,
        "optimization": "reward_weighted_sft",
    }

    run = wandb.init(project="wog-agent", job_type="policy_optimization", config=config)
    print(f"[wandb] {run.url}")

    # Log reward distribution
    log_reward_analysis(train_rewards, "train")
    log_reward_analysis(valid_rewards, "valid")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.base_model}")
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if not args.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config — higher rank than SFT for better policy expressivity
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    print("\nTokenizing datasets...")
    train_dataset = tokenize(train_records, train_weights, tokenizer, args.max_length)
    valid_dataset = tokenize(valid_records, valid_weights, tokenizer, args.max_length)

    # Training args
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        bf16=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = RewardWeightedTrainer(
        reward_weights=train_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting reward-weighted policy optimization...")
    result = trainer.train()
    print(f"  Train loss:   {result.training_loss:.4f}")
    print(f"  Runtime:      {result.metrics.get('train_runtime', 0):.1f}s")

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results.get("eval_loss", 0))
    print(f"  Val loss:     {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"  Perplexity:   {perplexity:.2f}")

    # Save adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}/")

    # Log as W&B artifact
    artifact = wandb.Artifact(
        "wog-policy-adapter",
        type="model",
        description=f"Reward-weighted policy adapter (rank={args.lora_rank}, {args.epochs} epochs)",
        metadata={**config, "final_train_loss": result.training_loss,
                  "final_val_loss": eval_results.get("eval_loss"),
                  "perplexity": perplexity},
    )
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            artifact.add_file(fpath)
    run.log_artifact(artifact)

    run.summary.update({
        "final_train_loss": result.training_loss,
        "final_val_loss": eval_results.get("eval_loss"),
        "perplexity": perplexity,
        "train_examples": len(train_records),
        "pct_positive_reward": config["pct_positive_reward"],
    })
    run.finish()
    print("\nPolicy optimization complete.")


if __name__ == "__main__":
    main()
