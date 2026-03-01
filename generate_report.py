"""
Generate a W&B Report for the WoG fine-tuning project.

Creates a structured report with per-model comparisons on:
  - Gameplay metrics: XP, kills, quests completed, gold, deaths
  - Quest performance by difficulty level
  - Tool-calling quality: validity, selection, argument completeness
  - Training curves
  - Inference time

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
    parser.add_argument("--title", default="WoG Agent: Per-Model Gameplay Report")
    args = parser.parse_args()

    entity = args.entity or wandb.Api().default_entity

    report = wr.Report(
        project=args.project,
        entity=entity,
        title=args.title,
        description=(
            "Per-model comparison of gameplay performance (XP, kills, quests completed), "
            "quest difficulty breakdown, tool-calling quality, and training metrics."
        ),
    )

    report.blocks = [
        # ── Overview ──
        wr.H1("Overview"),
        wr.MarkdownBlock(
            "This report compares model performance across gameplay and tool-calling metrics.\n\n"
            "**Models compared:**\n"
            "- **Base**: Hermes-2-Pro-Mistral-7B (8-bit, prompt-engineered)\n"
            "- **Fine-tuned**: Base + LoRA adapter trained on self-play trajectories\n\n"
            "**Key metrics per model:**\n"
            "- XP earned (total and per cycle)\n"
            "- Number of kills (total and per cycle)\n"
            "- Number of completed quests (total and per cycle)\n"
            "- Deaths, gold earned\n"
            "- Quests completed per difficulty level\n"
            "- Inference time\n"
        ),

        # ── Gameplay: XP ──
        wr.H1("XP Earned"),
        wr.MarkdownBlock(
            "Total XP earned over time, per model run. Higher is better."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["gameplay/total_xp"], title="Total XP (cumulative)"),
                wr.BarPlot(
                    metrics=[wr.metrics.SummaryMetric("gameplay/total_xp")],
                    groupby=wr.metrics.Config("model"),
                    title="Total XP by Model",
                ),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity),
            ],
        ),

        # ── Gameplay: Kills ──
        wr.H1("Kills"),
        wr.MarkdownBlock(
            "Total mobs killed over time, per model run. Higher is better."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["gameplay/total_kills"], title="Total Kills (cumulative)"),
                wr.BarPlot(
                    metrics=[wr.metrics.SummaryMetric("gameplay/total_kills")],
                    groupby=wr.metrics.Config("model"),
                    title="Total Kills by Model",
                ),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity),
            ],
        ),

        # ── Gameplay: Quests Completed ──
        wr.H1("Quests Completed"),
        wr.MarkdownBlock(
            "Total quests completed over time. This is the highest-weighted "
            "metric in our reward function (50x weight)."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["gameplay/quests_completed"], title="Quests Completed (cumulative)"),
                wr.BarPlot(
                    metrics=[wr.metrics.SummaryMetric("gameplay/quests_completed")],
                    groupby=wr.metrics.Config("model"),
                    title="Quests Completed by Model",
                ),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity),
            ],
        ),

        # ── Quest Performance by Difficulty ──
        wr.H1("Quest Performance by Difficulty"),
        wr.MarkdownBlock(
            "Breakdown of quests completed, XP earned, and average completion time "
            "per difficulty level. Tracked from the evaluation run."
        ),
        wr.PanelGrid(
            panels=[
                wr.RunComparer(diff_only="split"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters="JobType = 'evaluation'"),
            ],
        ),

        # ── Deaths & Efficiency ──
        wr.H1("Deaths & Efficiency"),
        wr.MarkdownBlock(
            "Fewer deaths = better strategy. Kill/death ratio and reward per cycle "
            "measure overall efficiency."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["gameplay/total_deaths"], title="Total Deaths"),
                wr.LinePlot(x="Step", y=["gameplay/total_gold_earned"], title="Gold Earned"),
                wr.LinePlot(x="Step", y=["quests/avg_completion_time_s"], title="Avg Quest Time (s)"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity),
            ],
        ),

        # ── Inference Time ──
        wr.H1("Inference Time"),
        wr.MarkdownBlock(
            "Average inference time per cycle. LoRA adapters add minimal overhead "
            "on top of the base model."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["system/inference_time_s"], title="Inference Time (s)"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity),
            ],
        ),

        # ── GPU Metrics (NVIDIA runs only) ──
        wr.H1("GPU Memory (NVIDIA)"),
        wr.MarkdownBlock(
            "CUDA GPU memory usage during the production run. Sampled every 10 seconds by W&B.\n\n"
            "- **Allocated**: memory actively used by tensors\n"
            "- **Reserved**: memory held by PyTorch allocator (includes cached blocks)\n"
            "- **Utilization %**: allocated / total device memory\n\n"
            "Filtered to runs with `config.backend = 'nvidia'`."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(
                    x="Step",
                    y=["gpu/0/memory_allocated_gb"],
                    title="GPU 0 — Allocated Memory (GB)",
                ),
                wr.LinePlot(
                    x="Step",
                    y=["gpu/0/memory_reserved_gb"],
                    title="GPU 0 — Reserved Memory (GB)",
                ),
                wr.LinePlot(
                    x="Step",
                    y=["gpu/0/memory_utilization_pct"],
                    title="GPU 0 — Memory Utilization (%)",
                ),
            ],
            runsets=[
                wr.Runset(
                    project=args.project,
                    entity=entity,
                    filters="config.backend = 'nvidia'",
                ),
            ],
        ),

        # ── Tool-Calling Quality ──
        wr.H1("Tool-Calling Quality"),
        wr.MarkdownBlock(
            "Offline evaluation on held-out validation data.\n\n"
            "| Metric | What it measures |\n"
            "|--------|------------------|\n"
            "| Tool Call Valid Rate | Did the model produce parseable JSON? |\n"
            "| Tool Selection Accuracy | Did it pick the right tool? |\n"
            "| Argument Completeness | Did it include all expected arguments? |\n"
            "| Tool Tags Rate | Did it use `<tool_call>` formatting? |"
        ),
        wr.PanelGrid(
            panels=[
                wr.RunComparer(diff_only="split"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters="JobType = 'evaluation'"),
            ],
        ),

        # ── Training Curves ──
        wr.H1("Training Curves"),
        wr.MarkdownBlock(
            "LoRA fine-tuning loss curves. Lower validation loss = better generalization."
        ),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x="Step", y=["train/loss"], title="Training Loss"),
                wr.LinePlot(x="Step", y=["val/loss"], title="Validation Loss"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters="JobType = 'training'"),
            ],
        ),

        # ── Data Collection ──
        wr.H1("Data Collection"),
        wr.MarkdownBlock(
            "### Trajectory Logging\n"
            "Every game loop cycle is captured with:\n"
            "- Full ChatML prompt and model response\n"
            "- Tool call + MCP result\n"
            "- Reward signals: gold, XP, kills, deaths, quests, zones\n"
            "- Composite scalar reward (same weights as policy optimizer)\n"
            "- Quest completion time and difficulty\n\n"
            "### Filtering for Training\n"
            "- `tool_success == True`\n"
            "- `deaths_delta == 0`\n"
            "- `reward >= threshold` (configurable)\n"
            "- Info-gathering tools (scans, status checks) kept at reward=0"
        ),
        wr.PanelGrid(
            panels=[
                wr.RunComparer(diff_only="split"),
            ],
            runsets=[
                wr.Runset(project=args.project, entity=entity, filters="JobType = 'data-prep'"),
            ],
        ),

        # ── Reward Function ──
        wr.H1("Reward Function"),
        wr.MarkdownBlock(
            "The composite reward used for trajectory filtering and policy optimization:\n\n"
            "```\n"
            "reward = gold * 3.0\n"
            "       + quests_completed * 50.0\n"
            "       + xp * 0.1\n"
            "       + deaths * (-10.0)\n"
            "       + zones_discovered * 5.0\n"
            "       + quest_gold * 3.0\n"
            "       + quest_xp * 0.1\n"
            "```\n\n"
            "Quest completions are weighted 50x because they represent multi-step "
            "reasoning (navigate, fight, gather, return). Deaths are penalized heavily."
        ),
    ]

    report.save()
    print(f"\nReport created: {report.url}")


if __name__ == "__main__":
    main()
