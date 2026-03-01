"""
Modal runner for WoG Agent experiments on H100 / A100.

Functions:
  run_agent     — 24/7 game loop for one agent instance (H100)
  run_training  — LoRA fine-tuning on collected data (H100)
  run_eval      — base vs fine-tuned evaluation (A100)

Spin up 4 parallel agents:
  modal run modal_runner.py          (default: 4 agents)
  modal run modal_runner.py::run_agent --agent-id 0

Run training:
  modal run modal_runner.py::run_training

Run eval:
  modal run modal_runner.py::run_eval

Setup (one-time):
  modal secret create wandb-secret WANDB_API_KEY=<your-key>

Persistent volume keeps per-agent wallet + memory alive across runs.
  /data/agent_0/.wallet_key_0   /data/agent_0/.memory_0.json
  /data/agent_1/.wallet_key_1   /data/agent_1/.memory_1.json
  ...
"""

import os
import shutil
import subprocess
import time

import modal

# ── Image ──────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "rsync")
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
        "firebase-admin",
    )
)

app = modal.App("wog-agent", image=image)

# Persistent volume: per-agent wallet, memory, trajectories, adapters
volume = modal.Volume.from_name("wog-agent-data", create_if_missing=True)

NUM_AGENTS = 8


def _agent_persist_files(agent_id: int) -> list[str]:
    return [f".wallet_key_{agent_id}", f".memory_{agent_id}.json"]


def _restore(agent_id: int, src_dir: str = "/data", dst_dir: str = "/app") -> None:
    """Copy persisted files for this agent from volume into the working dir."""
    agent_src = os.path.join(src_dir, f"agent_{agent_id}")
    for f in _agent_persist_files(agent_id):
        src = os.path.join(agent_src, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[volume] agent-{agent_id} restored {f}")


def _save(agent_id: int, src_dir: str = "/app", dst_dir: str = "/data") -> None:
    """Copy working files for this agent back to the volume."""
    agent_dst = os.path.join(dst_dir, f"agent_{agent_id}")
    os.makedirs(agent_dst, exist_ok=True)
    for f in _agent_persist_files(agent_id):
        src = os.path.join(src_dir, f)
        dst = os.path.join(agent_dst, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[volume] agent-{agent_id} saved {f}")
    # Sync trajectory data under agent-specific subdir
    traj_src = os.path.join(src_dir, "data", "raw")
    traj_dst = os.path.join(agent_dst, "data", "raw")
    if os.path.exists(traj_src):
        os.makedirs(traj_dst, exist_ok=True)
        subprocess.run(["rsync", "-a", traj_src + "/", traj_dst + "/"], check=False)
        print(f"[volume] agent-{agent_id} synced trajectories")
    volume.commit()


# ── Agent run ───────────────────────────────────────────────────────────────
@app.function(
    gpu="H100",
    timeout=86400,          # 24 hours
    volumes={"/data": volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("wog-firebase"),
    ],
)
def run_agent(agent_id: int = 0):
    """Run one WoG quest-chaining agent on H100."""
    import torch
    print(f"[agent-{agent_id}] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[agent-{agent_id}] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    subprocess.run(["git", "clone",
        "https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git", "/app"], check=True)
    os.chdir("/app")
    _restore(agent_id)

    env = {**os.environ, "AGENT_ID": str(agent_id)}
    proc = subprocess.Popen(
        ["python", "app_nvidia.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
        env=env,
    )

    SYNC_INTERVAL = 1800  # sync trajectories to volume every 30 minutes
    last_sync = time.time()

    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            print(f"[agent-{agent_id}] {line}", end="", flush=True)
            now = time.time()
            if now - last_sync >= SYNC_INTERVAL:
                _save(agent_id)
                print(f"[agent-{agent_id}] [volume] periodic sync done")
                last_sync = now
    finally:
        proc.wait()
        _save(agent_id)


# ── LoRA training ───────────────────────────────────────────────────────────
@app.function(
    gpu="H100",
    timeout=43200,          # 12 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("wog-firebase")],
)
def run_training(epochs: int = 3, lr: float = 1e-5, lora_rank: int = 8):
    """Fine-tune with LoRA on collected trajectories from all agents."""
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    subprocess.run(["git", "clone",
        "https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git", "/app"], check=True)
    os.chdir("/app")

    # Copy pre-merged train/valid data from volume root
    os.makedirs("/app/data", exist_ok=True)
    for fname in ["train.jsonl", "valid.jsonl"]:
        src = f"/data/{fname}"
        dst = f"/app/data/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[volume] copied {fname} ({os.path.getsize(dst)} bytes)")

    proc = subprocess.Popen(
        [
            "python", "train_lora_nvidia.py",
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--lora-rank", str(lora_rank),
            "--load-in-4bit",
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
        adapter_src = "/app/adapters"
        adapter_dst = "/data/adapters"
        if os.path.exists(adapter_src):
            subprocess.run(
                ["rsync", "-a", adapter_src + "/", adapter_dst + "/"], check=False
            )
            volume.commit()
            print("[volume] saved adapter")

            # Upload adapter files to Firebase Storage
            try:
                import json as _json
                import firebase_admin
                from firebase_admin import credentials, storage as fb_storage
                sa = _json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(
                        credentials.Certificate(sa),
                        {"storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"]},
                    )
                bucket = fb_storage.bucket()
                for root, _, files in os.walk(adapter_src):
                    for fname in files:
                        local_path = os.path.join(root, fname)
                        rel = os.path.relpath(local_path, adapter_src)
                        blob_path = f"adapters/{rel}"
                        bucket.blob(blob_path).upload_from_filename(local_path)
                        print(f"[firebase] uploaded adapters/{rel}")
                print("[firebase] Adapter fully saved to Firebase Storage")
            except Exception as e:
                print(f"[firebase] Upload failed (adapter safe in volume): {e}")


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
    """Spin up 4 quest-chaining agents in parallel on H100s."""
    print(f"Spawning {NUM_AGENTS} agents in parallel...")
    # starmap dispatches all 4 to separate H100 containers simultaneously
    list(run_agent.starmap([[i] for i in range(NUM_AGENTS)]))
