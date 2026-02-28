"""
LoRA fine-tuning script for the WoG agent.

Uses mlx_lm.lora to fine-tune Hermes-2-Pro-Mistral-7B on filtered
self-play trajectories. Logs training metrics to W&B in real-time
and saves the adapter as a W&B Artifact + pushes to HuggingFace Hub.

Usage:
    python train_lora.py [--base-model mlx-community/Hermes-2-Pro-Mistral-7B-8bit]
                         [--data data] [--iters 1000] [--lr 1e-5]
                         [--hf-repo YOUR_USERNAME/wog-lora-adapter]
"""

import argparse
import json
import os
import re
import subprocess
import sys

import wandb
from huggingface_hub import HfApi


def parse_loss_lines(line: str) -> dict | None:
    """Parse mlx_lm.lora stdout for loss values.

    Training lines look like: "Iter 10: Train loss 2.345, Learning Rate 1.000e-05"
    Validation lines look like: "Iter 10: Val loss 2.123"
    """
    train_match = re.search(
        r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)",
        line,
    )
    if train_match:
        return {
            "iter": int(train_match.group(1)),
            "train_loss": float(train_match.group(2)),
        }

    val_match = re.search(
        r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)",
        line,
    )
    if val_match:
        return {
            "iter": int(val_match.group(1)),
            "val_loss": float(val_match.group(2)),
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune WoG agent")
    parser.add_argument("--base-model", default="mlx-community/Hermes-2-Pro-Mistral-7B-8bit")
    parser.add_argument("--data", default="data", help="Directory with train.jsonl and valid.jsonl")
    parser.add_argument("--adapter-path", default="adapters", help="Output directory for LoRA adapter")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--val-batches", type=int, default=25, help="Number of validation batches")
    parser.add_argument("--steps-per-eval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--steps-per-report", type=int, default=10, help="Steps between loss reports")
    parser.add_argument("--save-every", type=int, default=200, help="Save adapter every N steps")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo to push adapter to")
    args = parser.parse_args()

    config = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "learning_rate": args.lr,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "val_batches": args.val_batches,
        "steps_per_eval": args.steps_per_eval,
    }

    # Count training examples
    train_path = os.path.join(args.data, "train.jsonl")
    valid_path = os.path.join(args.data, "valid.jsonl")
    for path in [train_path, valid_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run prepare_training_data.py first.")
            sys.exit(1)

    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(valid_path) as f:
        valid_count = sum(1 for _ in f)
    config["train_examples"] = train_count
    config["valid_examples"] = valid_count

    print(f"Training data: {train_count} train, {valid_count} valid examples")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Init W&B
    run = wandb.init(
        project="wog-agent",
        job_type="training",
        config=config,
    )
    print(f"W&B run: {run.url}")

    # Log training data as input artifact
    try:
        run.use_artifact("wog-training-data:latest")
    except Exception:
        pass  # Artifact may not exist if data prep didn't log it

    # Build mlx_lm.lora command with YAML config for LoRA params
    os.makedirs(args.adapter_path, exist_ok=True)

    config_path = os.path.join(args.adapter_path, "lora_config.yaml")
    with open(config_path, "w") as f:
        f.write(
            f"lora_parameters:\n"
            f"  rank: {args.lora_rank}\n"
            f"  alpha: {args.lora_rank * 2}\n"
            f"  dropout: 0.0\n"
            f"  scale: 1.0\n"
        )

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base_model,
        "--train",
        "--data", args.data,
        "--adapter-path", args.adapter_path,
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--num-layers", str(args.lora_layers),
        "--batch-size", str(args.batch_size),
        "--val-batches", str(args.val_batches),
        "--steps-per-eval", str(args.steps_per_eval),
        "--steps-per-report", str(args.steps_per_report),
        "--save-every", str(args.save_every),
        "-c", config_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run training and parse output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    best_val_loss = float("inf")

    for line in process.stdout or []:
        line = line.rstrip()
        print(line)

        parsed = parse_loss_lines(line)
        if parsed:
            step = parsed["iter"]
            metrics = {}
            if "train_loss" in parsed:
                metrics["train/loss"] = parsed["train_loss"]
            if "val_loss" in parsed:
                metrics["val/loss"] = parsed["val_loss"]
                if parsed["val_loss"] < best_val_loss:
                    best_val_loss = parsed["val_loss"]
                    metrics["val/best_loss"] = best_val_loss
            if metrics:
                run.log(metrics, step=step)

    process.wait()
    if process.returncode != 0:
        print(f"\nERROR: Training failed with return code {process.returncode}")
        run.finish(exit_code=1)
        sys.exit(1)

    print(f"\nTraining complete! Adapter saved to {args.adapter_path}/")

    # Log adapter as W&B Artifact
    artifact = wandb.Artifact(
        "wog-lora-adapter",
        type="model",
        description=f"LoRA adapter (rank={args.lora_rank}, {args.iters} iters) for WoG agent",
        metadata={
            **config,
            "best_val_loss": best_val_loss,
        },
    )
    for fname in os.listdir(args.adapter_path):
        fpath = os.path.join(args.adapter_path, fname)
        if os.path.isfile(fpath):
            artifact.add_file(fpath)
    run.log_artifact(artifact)
    print("LoRA adapter logged as W&B Artifact.")

    # Push to HuggingFace Hub
    if args.hf_repo:
        print(f"\nPushing adapter to HuggingFace Hub: {args.hf_repo}")
        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True)
        api.upload_folder(
            folder_path=args.adapter_path,
            repo_id=args.hf_repo,
            commit_message=f"WoG LoRA adapter (rank={args.lora_rank}, {args.iters} iters, val_loss={best_val_loss:.4f})",
        )
        print(f"Adapter pushed to https://huggingface.co/{args.hf_repo}")

    run.summary["best_val_loss"] = best_val_loss
    run.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
