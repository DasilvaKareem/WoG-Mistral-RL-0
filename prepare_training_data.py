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


def filter_trajectories(records: list[dict], reward_threshold: float = 0.0) -> list[dict]:
    """Keep only cycles where:
    - tool_success is True
    - tool_name is not None (model actually called a tool)
    - deaths_delta == 0 (no deaths in this cycle)
    - reward >= reward_threshold (positive-outcome filtering for RL)

    Records with reward > 0 are always kept (productive actions).
    Records with reward == 0 are kept only if they are status checks or scans
    (information-gathering is useful even with no immediate reward).
    """
    filtered = []
    info_tools = {"get_my_status", "scan_zone", "quests_list", "quests_progress", "inventory", "look_around"}
    for r in records:
        if not r.get("tool_success"):
            continue
        if not r.get("tool_name"):
            continue
        rewards = r.get("reward_signals", {})
        if rewards.get("deaths_delta", 0) > 0:
            continue
        reward = r.get("reward", 0.0)
        # Keep productive actions and info-gathering tools
        if reward >= reward_threshold or r.get("tool_name") in info_tools:
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
    zone_counts = Counter()
    difficulty_counts = Counter()
    difficulty_xp: dict[str, float] = {}
    difficulty_times: dict[str, list[float]] = {}
    reward_sums = {"gold": 0, "xp": 0, "kills": 0, "quests": 0, "quest_gold": 0,
                   "quest_xp": 0, "zones_discovered": 0, "zone_transitions": 0}
    rewards_list = []
    inference_by_difficulty: dict[str, list[float]] = {}
    for r in filtered:
        tool_counts[r.get("tool_name", "unknown")] += 1
        signals = r.get("reward_signals", {})
        reward_sums["gold"] += signals.get("gold_delta", 0)
        reward_sums["xp"] += signals.get("xp_delta", 0)
        reward_sums["kills"] += signals.get("kills_delta", 0)
        reward_sums["quests"] += signals.get("quests_completed_delta", 0)
        reward_sums["quest_gold"] += signals.get("quest_gold_delta", 0)
        reward_sums["quest_xp"] += signals.get("quest_xp_delta", 0)
        reward_sums["zones_discovered"] += signals.get("zones_discovered_delta", 0)
        reward_sums["zone_transitions"] += signals.get("zone_transitions_delta", 0)
        rewards_list.append(r.get("reward", 0.0))
        zone = signals.get("zone_after") or signals.get("zone_before")
        if zone:
            zone_counts[zone] += 1

        # Per-difficulty stats for quest completions
        diff = signals.get("quest_difficulty")
        if diff and signals.get("quests_completed_delta", 0) > 0:
            diff = str(diff)
            difficulty_counts[diff] += 1
            difficulty_xp.setdefault(diff, 0.0)
            difficulty_xp[diff] += signals.get("quest_xp_delta", 0)
            qt = signals.get("quest_completion_time_s")
            if qt is not None:
                difficulty_times.setdefault(diff, []).append(qt)
            inf_time = r.get("inference_time")
            if inf_time is not None:
                inference_by_difficulty.setdefault(diff, []).append(inf_time)

    rewards_list.sort()
    n = len(rewards_list)

    return {
        "raw_count": len(raw),
        "filtered_count": len(filtered),
        "formatted_count": len(formatted),
        "train_count": len(train),
        "valid_count": len(valid),
        "filter_rate": len(filtered) / max(len(raw), 1),
        "tool_distribution": dict(tool_counts.most_common(20)),
        "zone_distribution": dict(zone_counts.most_common(20)),
        # Reward distribution
        "reward_mean": sum(rewards_list) / max(n, 1),
        "reward_median": rewards_list[n // 2] if n else 0,
        "reward_min": rewards_list[0] if n else 0,
        "reward_max": rewards_list[-1] if n else 0,
        "reward_positive_pct": sum(1 for r in rewards_list if r > 0) / max(n, 1),
        # Totals in training data
        "total_gold_in_data": reward_sums["gold"],
        "total_xp_in_data": reward_sums["xp"],
        "total_kills_in_data": reward_sums["kills"],
        "total_quests_in_data": reward_sums["quests"],
        "total_quest_gold_in_data": reward_sums["quest_gold"],
        "total_quest_xp_in_data": reward_sums["quest_xp"],
        "total_zones_discovered_in_data": reward_sums["zones_discovered"],
        "total_zone_transitions_in_data": reward_sums["zone_transitions"],
        # Per-difficulty breakdown
        "quests_by_difficulty": dict(difficulty_counts.most_common()),
        "xp_by_difficulty": difficulty_xp,
        "avg_time_by_difficulty": {
            d: sum(t) / len(t) for d, t in difficulty_times.items()
        },
        "avg_inference_by_difficulty": {
            d: sum(t) / len(t) for d, t in inference_by_difficulty.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare LoRA training data from trajectories")
    parser.add_argument("--input", default="data/raw", help="Directory with raw trajectory JSONL files")
    parser.add_argument("--output", default="data", help="Output directory for train/valid JSONL")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--reward-threshold", type=float, default=0.0,
                        help="Minimum composite reward to include (default 0.0)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and filter
    print(f"Loading trajectories from {args.input}...")
    raw = load_raw_trajectories(args.input)
    print(f"  Raw records: {len(raw)}")

    filtered = filter_trajectories(raw, reward_threshold=args.reward_threshold)
    print(f"  After filtering (threshold={args.reward_threshold}): "
          f"{len(filtered)} ({len(filtered)/max(len(raw),1)*100:.1f}%)")

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

    # Log tool distribution chart
    tool_table = wandb.Table(
        columns=["tool", "count"],
        data=[[k, v] for k, v in stats["tool_distribution"].items()],
    )
    run.log({"tool_distribution": wandb.plot.bar(tool_table, "tool", "count", title="Tool Distribution in Training Data")})

    # Log zone distribution chart
    if stats.get("zone_distribution"):
        zone_table = wandb.Table(
            columns=["zone", "count"],
            data=[[k, v] for k, v in stats["zone_distribution"].items()],
        )
        run.log({"zone_distribution": wandb.plot.bar(zone_table, "zone", "count", title="Zone Distribution in Training Data")})

    # Log reward distribution histogram
    reward_values = [r.get("reward", 0.0) for r in filtered]
    if reward_values:
        reward_table = wandb.Table(columns=["reward"], data=[[r] for r in reward_values])
        run.log({"reward_distribution": wandb.plot.histogram(reward_table, "reward", title="Reward Distribution")})

    run.finish()
    print("\nDone! W&B artifact logged.")


if __name__ == "__main__":
    main()
