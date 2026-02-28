"""
Prepare training data for LoRA fine-tuning.

Reads raw JSONL trajectories, filters for successful tool-calling cycles,
formats into MLX LoRA chat format, and creates 80/20 train/valid split.
Logs dataset stats as a W&B Artifact.

Usage:
    python prepare_training_data.py [--input data/raw] [--output data]
"""

import argparse
import glob
import json
import os
import random
from collections import Counter

import wandb


def load_raw_trajectories(input_dir: str) -> list[dict]:
    """Load all JSONL trajectory files from input_dir."""
    records = []
    for path in sorted(glob.glob(os.path.join(input_dir, "traj_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def filter_trajectories(records: list[dict]) -> list[dict]:
    """Keep only cycles where:
    - tool_success is True
    - tool_name is not None (model actually called a tool)
    - deaths_delta == 0 (no deaths in this cycle)
    """
    filtered = []
    for r in records:
        if not r.get("tool_success"):
            continue
        if not r.get("tool_name"):
            continue
        rewards = r.get("reward_signals", {})
        if rewards.get("deaths_delta", 0) > 0:
            continue
        filtered.append(r)
    return filtered


def format_as_chat(record: dict) -> dict | None:
    """Convert a trajectory record into MLX LoRA chat format.

    Format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    The assistant message is the raw model response containing the <tool_call> block.
    """
    messages = record.get("messages", [])

    # Extract system message
    system_msg = None
    user_msgs = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg
        else:
            user_msgs.append(msg)

    if system_msg is None:
        return None

    # Take the last user message as the immediate context
    last_user = None
    for msg in reversed(user_msgs):
        if msg["role"] == "user":
            last_user = msg
            break

    if last_user is None:
        return None

    # The assistant response should contain the tool call
    response = record.get("response", "")
    if not response.strip():
        return None

    chat = {
        "messages": [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": last_user["content"]},
            {"role": "assistant", "content": response},
        ]
    }
    return chat


def compute_stats(
    raw: list[dict],
    filtered: list[dict],
    formatted: list[dict],
    train: list[dict],
    valid: list[dict],
) -> dict:
    """Compute dataset statistics for logging."""
    tool_counts = Counter()
    reward_sums = {"gold": 0, "xp": 0, "kills": 0}
    for r in filtered:
        tool_counts[r.get("tool_name", "unknown")] += 1
        rewards = r.get("reward_signals", {})
        reward_sums["gold"] += rewards.get("gold_delta", 0)
        reward_sums["xp"] += rewards.get("xp_delta", 0)
        reward_sums["kills"] += rewards.get("kills_delta", 0)

    return {
        "raw_count": len(raw),
        "filtered_count": len(filtered),
        "formatted_count": len(formatted),
        "train_count": len(train),
        "valid_count": len(valid),
        "filter_rate": len(filtered) / max(len(raw), 1),
        "tool_distribution": dict(tool_counts.most_common(20)),
        "total_gold_in_data": reward_sums["gold"],
        "total_xp_in_data": reward_sums["xp"],
        "total_kills_in_data": reward_sums["kills"],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare LoRA training data from trajectories")
    parser.add_argument("--input", default="data/raw", help="Directory with raw trajectory JSONL files")
    parser.add_argument("--output", default="data", help="Output directory for train/valid JSONL")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and filter
    print(f"Loading trajectories from {args.input}...")
    raw = load_raw_trajectories(args.input)
    print(f"  Raw records: {len(raw)}")

    filtered = filter_trajectories(raw)
    print(f"  After filtering: {len(filtered)} ({len(filtered)/max(len(raw),1)*100:.1f}%)")

    # Format as chat
    formatted = []
    for r in filtered:
        chat = format_as_chat(r)
        if chat is not None:
            formatted.append(chat)
    print(f"  Formatted examples: {len(formatted)}")

    if not formatted:
        print("ERROR: No valid training examples after filtering. Collect more data.")
        return

    # Shuffle and split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * args.split)
    train = formatted[:split_idx]
    valid = formatted[split_idx:]

    # Write output
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.jsonl")
    valid_path = os.path.join(args.output, "valid.jsonl")

    with open(train_path, "w") as f:
        for example in train:
            f.write(json.dumps(example) + "\n")

    with open(valid_path, "w") as f:
        for example in valid:
            f.write(json.dumps(example) + "\n")

    print(f"  Train: {len(train)} examples -> {train_path}")
    print(f"  Valid: {len(valid)} examples -> {valid_path}")

    # Log to W&B
    stats = compute_stats(raw, filtered, formatted, train, valid)
    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")

    run = wandb.init(
        project="wog-agent",
        job_type="data-prep",
        config={
            "input_dir": args.input,
            "split_ratio": args.split,
            "seed": args.seed,
            **stats,
        },
    )

    # Log as W&B Artifact
    artifact = wandb.Artifact(
        "wog-training-data",
        type="dataset",
        description=f"LoRA training data: {len(train)} train, {len(valid)} valid examples",
        metadata=stats,
    )
    artifact.add_file(train_path)
    artifact.add_file(valid_path)
    run.log_artifact(artifact)

    # Log tool distribution as a table
    tool_table = wandb.Table(
        columns=["tool", "count"],
        data=[[k, v] for k, v in stats["tool_distribution"].items()],
    )
    run.log({"tool_distribution": wandb.plot.bar(tool_table, "tool", "count", title="Tool Distribution in Training Data")})

    run.finish()
    print("\nDone! W&B artifact logged.")


if __name__ == "__main__":
    main()
