"""
Modal runner for WoG Agent experiments on H100 / A100.

Three functions:
  run_agent     — 24/7 game loop, collects trajectories (H100)
  run_training  — LoRA fine-tuning on collected data (H100)
  run_eval      — base vs fine-tuned evaluation (A100)

Persistent volume keeps wallet + memory alive across runs.

Setup (one-time):
  modal secret create wog-secrets \
    WANDB_API_KEY=<your-key> \
    HF_TOKEN=<optional, for pushing adapter>

Deploy agent:
  modal run modal_runner.py::run_agent

Deploy training:
  modal run modal_runner.py::run_training

Deploy eval:
  modal run modal_runner.py::run_eval
"""

import os
import shutil
import subprocess

import modal

# ── Image ──────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "mcp[cli]",
        "eth-account",
        "httpx",
        "wandb",
        "weave",
        "wandb-workspaces",
        "huggingface_hub",
        "trl",
    )
)


app = modal.App("wog-agent", image=image)

# Persistent volume: wallet key, memory, trajectory data, adapters
volume = modal.Volume.from_name("wog-agent-data", create_if_missing=True)

# Files to persist between runs
PERSIST = [".wallet_key", ".memory.json"]


def _restore(src_dir: str = "/data", dst_dir: str = "/app") -> None:
    """Copy persisted files from volume into the working dir."""
    for f in PERSIST:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[volume] restored {f}")


def _save(src_dir: str = "/app", dst_dir: str = "/data") -> None:
    """Copy working files back to the volume."""
    for f in PERSIST:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[volume] saved {f}")
    # Also sync trajectory data
    traj_src = os.path.join(src_dir, "data", "raw")
    traj_dst = os.path.join(dst_dir, "data", "raw")
    if os.path.exists(traj_src):
        os.makedirs(traj_dst, exist_ok=True)
        subprocess.run(["rsync", "-a", traj_src + "/", traj_dst + "/"], check=False)
        print("[volume] synced trajectory data")
    volume.commit()


# ── Agent run ───────────────────────────────────────────────────────────────
@app.function(
    gpu="H100",
    timeout=86400,          # 24 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_agent():
    """Run the WoG game agent on H100 — collects trajectories + W&B telemetry."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    subprocess.run(["git", "clone",
        "https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git", "/app"], check=True)
    os.chdir("/app")
    _restore()

    proc = subprocess.Popen(
        ["python", "app_nvidia.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
    )

    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            print(line, end="", flush=True)
    finally:
        proc.wait()
        _save()


# ── LoRA training ───────────────────────────────────────────────────────────
@app.function(
    gpu="H100",
    timeout=43200,          # 12 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_training(iters: int = 200, lr: float = 1e-5, lora_rank: int = 8):
    """Fine-tune with LoRA on collected trajectories."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    subprocess.run(["git", "clone",
        "https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git", "/app"], check=True)
    os.chdir("/app")

    # Pull latest trajectory data from volume
    traj_src = "/data/data/raw"
    traj_dst = "/app/data/raw"
    if os.path.exists(traj_src):
        os.makedirs(traj_dst, exist_ok=True)
        subprocess.run(["rsync", "-a", traj_src + "/", traj_dst + "/"], check=False)
        print("[volume] restored trajectory data")

    proc = subprocess.Popen(
        [
            "python", "train_lora_nvidia.py",
            "--iters", str(iters),
            "--lr", str(lr),
            "--lora-rank", str(lora_rank),
            "--data", "data",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
    )

    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            print(line, end="", flush=True)
    finally:
        proc.wait()
        # Save trained adapter back to volume
        adapter_src = "/app/adapters"
        adapter_dst = "/data/adapters"
        if os.path.exists(adapter_src):
            subprocess.run(
                ["rsync", "-a", adapter_src + "/", adapter_dst + "/"], check=False
            )
            volume.commit()
            print("[volume] saved adapter")


# ── Evaluation ──────────────────────────────────────────────────────────────
@app.function(
    gpu="A100-40GB",
    timeout=7200,           # 2 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_eval(max_examples: int = 50):
    """Evaluate base vs fine-tuned model on held-out validation data."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    subprocess.run(["git", "clone",
        "https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git", "/app"], check=True)
    os.chdir("/app")

    # Restore adapter from volume
    adapter_src = "/data/adapters"
    adapter_dst = "/app/adapters"
    if os.path.exists(adapter_src):
        os.makedirs(adapter_dst, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", adapter_src + "/", adapter_dst + "/"], check=False
        )
        print("[volume] restored adapter")

    proc = subprocess.Popen(
        [
            "python", "evaluate_nvidia.py",
            "--max-examples", str(max_examples),
            "--adapter-path", "adapters",
            "--data", "data/valid.jsonl",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
    )

    for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
        print(line, end="", flush=True)
    proc.wait()


# ── Local entrypoints ───────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Default: run the game agent."""
    run_agent.remote()
