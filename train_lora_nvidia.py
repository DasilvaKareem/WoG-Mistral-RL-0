"""
LoRA fine-tuning script for the WoG agent.
NVIDIA/CUDA version — uses PEFT + transformers Trainer.

Usage:
    python train_lora_nvidia.py [--base-model NousResearch/Hermes-2-Pro-Mistral-7B]
                                [--data data] [--iters 600] [--lr 1e-5]
                                [--hf-repo YOUR_USERNAME/wog-lora-adapter]
"""

import argparse
import json
import os
import sys

import torch
import wandb
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_jsonl_dataset(path: str) -> list[dict]:
    """Load JSONL chat dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_chatml(messages: list[dict]) -> str:
    """Convert messages list to ChatML string for training."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def tokenize_dataset(examples: list[dict], tokenizer, max_length: int = 2048) -> Dataset:
    """Tokenize chat examples into HuggingFace Dataset."""
    texts = []
    for ex in examples:
        messages = ex.get("messages", [])
        text = format_chatml(messages)
        texts.append(text)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    })
    return dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune WoG agent (NVIDIA/CUDA)")
    parser.add_argument("--base-model", default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--data", default="data", help="Directory with train.jsonl and valid.jsonl")
    parser.add_argument("--output-dir", default="adapters", help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo to push adapter to")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    args = parser.parse_args()

    # Verify data exists
    train_path = os.path.join(args.data, "train.jsonl")
    valid_path = os.path.join(args.data, "valid.jsonl")
    for path in [train_path, valid_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run prepare_training_data.py first.")
            sys.exit(1)

    # Load data
    print("Loading training data...")
    train_examples = load_jsonl_dataset(train_path)
    valid_examples = load_jsonl_dataset(valid_path)
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Valid: {len(valid_examples)} examples")

    # Config
    config = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "quantization": "4bit" if args.load_in_4bit else "8bit",
    }
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
        pass

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_examples, tokenizer, args.max_length)
    valid_dataset = tokenize_dataset(valid_examples, tokenizer, args.max_length)
    print(f"  Train tokens: {sum(len(ids) for ids in train_dataset['input_ids'])}")
    print(f"  Valid tokens: {sum(len(ids) for ids in valid_dataset['input_ids'])}")

    # Load model with quantization
    print(f"\nLoading model: {args.base_model}")
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    print("Applying LoRA adapter...")
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

    # Training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        fp16=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics.get('train_runtime', 0):.1f}s")

    # Save final adapter
    print(f"\nSaving adapter to {args.output_dir}/")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"  Val loss: {eval_results.get('eval_loss', 'N/A')}")

    # Log adapter as W&B Artifact
    artifact = wandb.Artifact(
        "wog-lora-adapter",
        type="model",
        description=f"LoRA adapter (rank={args.lora_rank}, {args.epochs} epochs) for WoG agent (NVIDIA)",
        metadata={
            **config,
            "final_train_loss": train_result.training_loss,
            "final_val_loss": eval_results.get("eval_loss"),
        },
    )
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
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
            folder_path=args.output_dir,
            repo_id=args.hf_repo,
            commit_message=f"WoG LoRA adapter (rank={args.lora_rank}, val_loss={eval_results.get('eval_loss', 'N/A')})",
        )
        print(f"Adapter pushed to https://huggingface.co/{args.hf_repo}")

    run.summary["final_train_loss"] = train_result.training_loss
    run.summary["final_val_loss"] = eval_results.get("eval_loss")
    run.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
