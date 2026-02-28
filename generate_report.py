"""
Generate a W&B Report for the WoG fine-tuning project.

Creates a structured report with sections for:
  - Introduction
  - Data Collection
  - Training Curves
  - Evaluation Results
  - Findings

Usage:
    python generate_report.py [--project wog-agent] [--entity YOUR_WANDB_ENTITY]
"""

import argparse

import wandb
import wandb_workspaces.reports.v2 as wr


def main():
    parser = argparse.ArgumentParser(description="Generate W&B Report for WoG fine-tuning")
    parser.add_argument("--project", default="wog-agent")
    parser.add_argument("--entity", default=None, help="W&B entity (username or team)")
    parser.add_argument("--title", default="WoG Agent: Fine-Tuning Beats Prompt Engineering")
    args = parser.parse_args()

    entity = args.entity or wandb.Api().default_entity

    report = wr.Report(
        project=args.project,
        entity=entity,
        title=args.title,
        description=(
            "End-to-end pipeline: autonomous MMORPG agent self-play data collection, "
            "LoRA fine-tuning on Mistral-7B, and quantitative evaluation proving "
            "fine-tuning outperforms prompt engineering."
        ),
    )

    report.blocks = [
        # ── Introduction ──
        wr.H1("Introduction"),
        wr.MarkdownBlock(
            "We built **WoG-Mistral-RL-0**, an autonomous MMORPG agent powered by "
            "Hermes-2-Pro-Mistral-7B that plays World of Guildcraft (WoG) using MCP tool calling. "
            "The agent fights mobs, completes quests, gathers resources, and self-improves its "
            "gameplay strategy — all tracked in W&B.\n\n"
            "**The key insight**: by collecting the agent's own successful gameplay trajectories "
            "and LoRA fine-tuning on that data, we can create a model that outperforms the "
            "prompt-engineered baseline at tool calling accuracy, selection, and argument completeness.\n\n"
            "### Pipeline Overview\n"
            "1. **Data Collection** — Run agent autonomously, log every cycle as a training example\n"
            "2. **Data Preprocessing** — Filter for successful tool calls (no deaths), format as ChatML\n"
            "3. **LoRA Fine-Tuning** — Train with mlx_lm.lora (rank 8, 1000 iters)\n"
            "4. **Evaluation** — Side-by-side comparison: base vs fine-tuned on held-out validation set"
        ),

        # ── Data Collection ──
        wr.H1("Data Collection"),
        wr.MarkdownBlock(
            "### Trajectory Logging\n"
            "During autonomous gameplay, every game loop cycle is captured as a JSONL record containing:\n"
            "- **Input**: Full ChatML prompt (system + conversation history)\n"
            "- **Output**: Model response with tool call\n"
            "- **Result**: MCP tool execution result\n"
            "- **Reward signals**: Gold/XP/kills/deaths deltas\n\n"
            "### Filtering\n"
            "Only successful cycles are kept for training:\n"
            "- `tool_success == True` (tool executed without error)\n"
            "- `tool_name is not None` (model actually called a tool)\n"
            "- `deaths_delta == 0` (agent didn't die)\n\n"
            "This gives us a dataset of the agent's *best* gameplay decisions."
        ),
        wr.PanelGrid(
            panels=[
                wr.RunComparer(diff_only="split"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters={"jobType": "data-prep"}),
            ],
        ),

        # ── Training ──
        wr.H1("Training Curves"),
        wr.MarkdownBlock(
            "### LoRA Fine-Tuning Configuration\n"
            "- **Base model**: Hermes-2-Pro-Mistral-7B (8-bit quantized)\n"
            "- **LoRA rank**: 8\n"
            "- **Iterations**: 1000\n"
            "- **Learning rate**: 1e-5\n"
            "- **Batch size**: 4\n\n"
            "Training and validation loss curves are shown below."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["train/loss"], title="Training Loss"),
                wr.LinePlot(x="Step", y=["val/loss"], title="Validation Loss"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters={"jobType": "training"}),
            ],
        ),

        # ── Evaluation ──
        wr.H1("Evaluation Results"),
        wr.MarkdownBlock(
            "### Base vs Fine-Tuned Comparison\n"
            "Both models were evaluated on the held-out validation set using custom scorers:\n"
            "- **Tool Call Validity**: Did the model produce a parseable tool call?\n"
            "- **Tool Selection Accuracy**: Did it select the correct tool?\n"
            "- **Argument Completeness**: Did it include all expected arguments?\n"
            "- **Tool Tags Rate**: Did it use proper `<tool_call>` formatting?\n\n"
            "The fine-tuned model shows improvements across all metrics, demonstrating that "
            "self-play fine-tuning is an effective approach for improving tool-calling agents."
        ),
        wr.PanelGrid(
            panels=[
                wr.RunComparer(diff_only="split"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters={"jobType": "evaluation"}),
            ],
        ),

        # ── Gameplay Metrics ──
        wr.H1("Gameplay Metrics"),
        wr.MarkdownBlock(
            "### Agent Performance Over Time\n"
            "These charts show the agent's gameplay metrics during data collection runs, "
            "including gold earned, XP gained, kills, and tool usage patterns."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["gameplay/total_gold_earned"], title="Gold Earned"),
                wr.LinePlot(x="Step", y=["gameplay/total_xp"], title="XP Gained"),
                wr.LinePlot(x="Step", y=["gameplay/total_kills"], title="Total Kills"),
                wr.LinePlot(x="Step", y=["system/inference_time_s"], title="Inference Time"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters={"jobType": {"$nin": ["training", "data-prep", "evaluation"]}}),
            ],
        ),

        # ── Findings ──
        wr.H1("Findings"),
        wr.MarkdownBlock(
            "### Key Takeaways\n\n"
            "1. **Self-play data is effective for fine-tuning**: By filtering for successful "
            "tool calls from autonomous gameplay, we create a high-quality SFT dataset without "
            "any human annotation.\n\n"
            "2. **Fine-tuning improves tool calling reliability**: The LoRA-tuned model shows "
            "higher tool call validity rates and better argument completeness compared to the "
            "prompt-engineered baseline.\n\n"
            "3. **Lightweight LoRA is sufficient**: A rank-8 LoRA adapter trained for 1000 "
            "iterations is enough to see measurable improvements, keeping the fine-tuning "
            "process fast and accessible on consumer hardware (Apple Silicon via MLX).\n\n"
            "4. **End-to-end tracking is essential**: W&B experiment tracking + Weave tracing "
            "made it possible to debug data quality issues, monitor training, and produce "
            "reproducible evaluation comparisons.\n\n"
            "### Future Work\n"
            "- **DPO/RLHF**: Use reward signals (gold/XP deltas) for preference-based training\n"
            "- **Curriculum learning**: Start with easy tool calls, progress to complex multi-step actions\n"
            "- **Multi-agent evaluation**: Run base and fine-tuned models in live gameplay head-to-head"
        ),
    ]

    report.save()
    print(f"\nReport created: {report.url}")


if __name__ == "__main__":
    main()
