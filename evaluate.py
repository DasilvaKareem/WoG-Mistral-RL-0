"""
Evaluation script: base model vs LoRA fine-tuned model.

Evaluates on TWO axes:
  1. Tool-calling quality (offline, from validation data)
     - Tool call validity, selection accuracy, argument completeness
  2. Gameplay performance (from trajectory data per model)
     - XP earned, kills, quests completed, deaths, gold, inference time
     - Broken down by quest difficulty

Usage:
    python evaluate.py [--adapter-path adapters] [--data data/valid.jsonl]
                       [--trajectories data/raw]
"""

import argparse
import glob
import json
import os
import re
import asyncio
from collections import Counter, defaultdict
from typing import Any

import weave
from mlx_lm import load, generate


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call JSON block from model output."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
            if "name" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


class WoGAgent(weave.Model):
    """WoG agent model wrapper for Weave evaluation."""
    model_id: str = "mlx-community/Hermes-2-Pro-Mistral-7B-8bit"
    adapter_path: str | None = None
    label: str = "base"
    _model: Any = None
    _tokenizer: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def load_model(self):
        if self._model is None:
            result = load(self.model_id, adapter_path=self.adapter_path) if self.adapter_path else load(self.model_id)
            self._model, self._tokenizer = result[0], result[1]

    @weave.op()
    def predict(self, system_prompt: str, user_message: str) -> dict:
        self.load_model()
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        tool_call = parse_tool_call(response)
        return {
            "response": response,
            "tool_call": tool_call,
            "tool_name": tool_call.get("name") if tool_call else None,
            "tool_args": tool_call.get("arguments", {}) if tool_call else None,
            "has_tool_call": tool_call is not None,
        }


@weave.op()
def tool_call_validity(model_output: dict) -> dict:
    """Score: did the model produce a valid tool call?"""
    has_call = model_output.get("has_tool_call", False)
    tool_call = model_output.get("tool_call")

    valid = False
    if has_call and tool_call:
        has_name = bool(tool_call.get("name"))
        has_args = "arguments" in tool_call
        valid = has_name and has_args

    return {"tool_call_valid": valid}


@weave.op()
def tool_selection_accuracy(model_output: dict, expected_tool: str) -> dict:
    """Score: did the model select the correct tool?"""
    predicted = model_output.get("tool_name")
    correct = predicted == expected_tool if expected_tool else False
    return {"tool_selection_correct": correct}


@weave.op()
def argument_completeness(model_output: dict, expected_args: dict) -> dict:
    """Score: did the model include the expected argument keys?"""
    predicted_args = model_output.get("tool_args") or {}
    if not expected_args:
        return {"argument_completeness": 1.0}

    expected_keys = set(expected_args.keys()) - {"sessionId", "entityId", "zoneId"}
    if not expected_keys:
        return {"argument_completeness": 1.0}

    predicted_keys = set(predicted_args.keys()) - {"sessionId", "entityId", "zoneId"}
    overlap = expected_keys & predicted_keys
    completeness = len(overlap) / len(expected_keys)
    return {"argument_completeness": completeness}


@weave.op()
def response_quality(model_output: dict) -> dict:
    """Score: overall response quality heuristics."""
    response = model_output.get("response", "")
    scores = {
        "has_tool_tags": "<tool_call>" in response and "</tool_call>" in response,
        "response_length": len(response),
        "not_empty": len(response.strip()) > 0,
    }
    return scores


def load_eval_dataset(path: str) -> list[dict]:
    """Load validation data and extract eval examples."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]

            # Parse expected tool call from the assistant message
            expected_call = parse_tool_call(assistant_msg)

            examples.append({
                "system_prompt": system_msg,
                "user_message": user_msg,
                "expected_tool": expected_call.get("name") if expected_call else None,
                "expected_args": expected_call.get("arguments", {}) if expected_call else {},
                "expected_response": assistant_msg,
            })
    return examples


# ── Gameplay metrics from trajectory data ──

def load_trajectories(input_dir: str) -> list[dict]:
    """Load all JSONL trajectory files."""
    records = []
    for path in sorted(glob.glob(os.path.join(input_dir, "traj_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def compute_gameplay_metrics(trajectories: list[dict]) -> dict:
    """Aggregate gameplay metrics from trajectory records.

    Returns per-model stats on XP, kills, quests, deaths, gold, inference time,
    and per-difficulty breakdowns.
    """
    if not trajectories:
        return {}

    total_cycles = len(trajectories)
    signals_sum = defaultdict(float)
    tool_counts = Counter()
    inference_times = []
    difficulty_stats = defaultdict(lambda: {"count": 0, "xp": 0, "gold": 0, "times": []})

    for r in trajectories:
        signals = r.get("reward_signals", {})
        signals_sum["xp"] += signals.get("xp_delta", 0)
        signals_sum["gold"] += signals.get("gold_delta", 0)
        signals_sum["kills"] += signals.get("kills_delta", 0)
        signals_sum["deaths"] += signals.get("deaths_delta", 0)
        signals_sum["quests_completed"] += signals.get("quests_completed_delta", 0)
        signals_sum["quest_xp"] += signals.get("quest_xp_delta", 0)
        signals_sum["quest_gold"] += signals.get("quest_gold_delta", 0)
        signals_sum["zones_discovered"] += signals.get("zones_discovered_delta", 0)
        signals_sum["zone_transitions"] += signals.get("zone_transitions_delta", 0)
        signals_sum["reward"] += r.get("reward", 0.0)

        tool_counts[r.get("tool_name", "none")] += 1
        inf_t = r.get("inference_time")
        if inf_t is not None:
            inference_times.append(inf_t)

        # Per-difficulty
        diff = signals.get("quest_difficulty")
        if diff and signals.get("quests_completed_delta", 0) > 0:
            diff = str(diff)
            difficulty_stats[diff]["count"] += 1
            difficulty_stats[diff]["xp"] += signals.get("quest_xp_delta", 0)
            difficulty_stats[diff]["gold"] += signals.get("quest_gold_delta", 0)
            qt = signals.get("quest_completion_time_s")
            if qt is not None:
                difficulty_stats[diff]["times"].append(qt)

    avg_inf = sum(inference_times) / len(inference_times) if inference_times else 0

    return {
        "total_cycles": total_cycles,
        "total_xp": signals_sum["xp"],
        "total_kills": signals_sum["kills"],
        "total_deaths": signals_sum["deaths"],
        "total_gold": signals_sum["gold"],
        "total_quests_completed": signals_sum["quests_completed"],
        "total_quest_xp": signals_sum["quest_xp"],
        "total_quest_gold": signals_sum["quest_gold"],
        "total_zones_discovered": signals_sum["zones_discovered"],
        "total_reward": signals_sum["reward"],
        "xp_per_cycle": signals_sum["xp"] / max(total_cycles, 1),
        "kills_per_cycle": signals_sum["kills"] / max(total_cycles, 1),
        "gold_per_cycle": signals_sum["gold"] / max(total_cycles, 1),
        "quests_per_cycle": signals_sum["quests_completed"] / max(total_cycles, 1),
        "deaths_per_cycle": signals_sum["deaths"] / max(total_cycles, 1),
        "reward_per_cycle": signals_sum["reward"] / max(total_cycles, 1),
        "avg_inference_time_s": avg_inf,
        "tool_distribution": dict(tool_counts.most_common(15)),
        "quests_by_difficulty": {
            d: {
                "count": data["count"],
                "total_xp": data["xp"],
                "total_gold": data["gold"],
                "avg_time_s": sum(data["times"]) / len(data["times"]) if data["times"] else 0,
            }
            for d, data in sorted(difficulty_stats.items())
        },
    }


async def run_tool_eval(agent: WoGAgent, dataset: list[dict], label: str) -> dict:
    """Run tool-calling evaluation on dataset and collect scores."""
    results = {
        "label": label,
        "total": len(dataset),
        "tool_call_valid": 0,
        "tool_selection_correct": 0,
        "argument_completeness_sum": 0.0,
        "has_tool_tags": 0,
    }

    for i, example in enumerate(dataset):
        output = agent.predict(
            system_prompt=example["system_prompt"],
            user_message=example["user_message"],
        )

        validity = tool_call_validity(output)
        if validity["tool_call_valid"]:
            results["tool_call_valid"] += 1

        accuracy = tool_selection_accuracy(output, example["expected_tool"])
        if accuracy["tool_selection_correct"]:
            results["tool_selection_correct"] += 1

        completeness = argument_completeness(output, example["expected_args"])
        results["argument_completeness_sum"] += completeness["argument_completeness"]

        quality = response_quality(output)
        if quality["has_tool_tags"]:
            results["has_tool_tags"] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(dataset)} examples evaluated")

    n = max(results["total"], 1)
    results["tool_call_valid_rate"] = results["tool_call_valid"] / n
    results["tool_selection_accuracy"] = results["tool_selection_correct"] / n
    results["argument_completeness_avg"] = results["argument_completeness_sum"] / n
    results["has_tool_tags_rate"] = results["has_tool_tags"] / n

    return results


async def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned WoG agent")
    parser.add_argument("--base-model", default="mlx-community/Hermes-2-Pro-Mistral-7B-8bit")
    parser.add_argument("--adapter-path", default="adapters", help="Path to LoRA adapter")
    parser.add_argument("--data", default="data/valid.jsonl", help="Validation dataset")
    parser.add_argument("--trajectories", default="data/raw", help="Trajectory data directory")
    parser.add_argument("--max-examples", type=int, default=50, help="Max examples to evaluate")
    args = parser.parse_args()

    weave.init("wog-agent")

    # ── Part 1: Gameplay metrics from trajectories ──
    print("=" * 60)
    print("PART 1: GAMEPLAY METRICS (from trajectory data)")
    print("=" * 60)

    trajectories = load_trajectories(args.trajectories)
    if trajectories:
        gameplay = compute_gameplay_metrics(trajectories)
        print(f"\nTrajectory records: {gameplay['total_cycles']}")
        print(f"\n{'Metric':<30} {'Value':>12} {'Per Cycle':>12}")
        print("-" * 54)
        gp_metrics = [
            ("Total XP", "total_xp", "xp_per_cycle"),
            ("Total Kills", "total_kills", "kills_per_cycle"),
            ("Total Deaths", "total_deaths", "deaths_per_cycle"),
            ("Total Gold", "total_gold", "gold_per_cycle"),
            ("Quests Completed", "total_quests_completed", "quests_per_cycle"),
            ("Composite Reward", "total_reward", "reward_per_cycle"),
        ]
        for display, total_key, rate_key in gp_metrics:
            print(f"{display:<30} {gameplay[total_key]:>12.1f} {gameplay[rate_key]:>12.4f}")

        print(f"\nAvg Inference Time: {gameplay['avg_inference_time_s']:.3f}s")

        if gameplay["quests_by_difficulty"]:
            print(f"\n{'Difficulty':<12} {'Count':>8} {'XP':>8} {'Gold':>8} {'Avg Time':>10}")
            print("-" * 46)
            for diff, data in gameplay["quests_by_difficulty"].items():
                print(f"{diff:<12} {data['count']:>8} {data['total_xp']:>8.0f} "
                      f"{data['total_gold']:>8.0f} {data['avg_time_s']:>9.1f}s")
    else:
        gameplay = {}
        print("No trajectory data found. Run the agent first to collect data.")

    # ── Part 2: Tool-calling quality (offline eval) ──
    print(f"\n{'=' * 60}")
    print("PART 2: TOOL-CALLING QUALITY (offline eval)")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"No validation data at {args.data}. Skipping tool eval.")
        tool_results = {}
    else:
        dataset = load_eval_dataset(args.data)
        if len(dataset) > args.max_examples:
            dataset = dataset[:args.max_examples]
        print(f"Evaluating on {len(dataset)} examples\n")

        print(f"Loading base model: {args.base_model}")
        base_agent = WoGAgent(model_id=args.base_model, label="base")
        base_agent.load_model()

        print(f"Loading fine-tuned model: {args.base_model} + {args.adapter_path}")
        ft_agent = WoGAgent(
            model_id=args.base_model,
            adapter_path=args.adapter_path,
            label="fine-tuned",
        )
        ft_agent.load_model()

        print("\nEvaluating base model...")
        base_results = await run_tool_eval(base_agent, dataset, "base")

        print("\nEvaluating fine-tuned model...")
        ft_results = await run_tool_eval(ft_agent, dataset, "fine-tuned")

        print(f"\n{'Metric':<30} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print("-" * 62)

        tool_metrics = [
            ("Tool Call Valid Rate", "tool_call_valid_rate"),
            ("Tool Selection Accuracy", "tool_selection_accuracy"),
            ("Argument Completeness", "argument_completeness_avg"),
            ("Has Tool Tags Rate", "has_tool_tags_rate"),
        ]

        for display_name, key in tool_metrics:
            base_val = base_results[key]
            ft_val = ft_results[key]
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"{display_name:<30} {base_val:>10.3f} {ft_val:>12.3f} {sign}{delta:>9.3f}")

        tool_results = {"base": base_results, "fine_tuned": ft_results}

    # ── Log everything to W&B ──
    print(f"\n{'=' * 60}")
    print("LOGGING TO W&B")
    print("=" * 60)

    import wandb
    run = wandb.init(project="wog-agent", job_type="evaluation")

    # Gameplay metrics
    if gameplay:
        for key in ["total_xp", "total_kills", "total_deaths", "total_gold",
                     "total_quests_completed", "total_reward", "avg_inference_time_s",
                     "xp_per_cycle", "kills_per_cycle", "gold_per_cycle",
                     "quests_per_cycle", "deaths_per_cycle", "reward_per_cycle"]:
            run.summary[f"gameplay/{key}"] = gameplay.get(key, 0)

        # Per-difficulty table
        if gameplay.get("quests_by_difficulty"):
            diff_table = wandb.Table(
                columns=["difficulty", "count", "total_xp", "total_gold", "avg_time_s"],
                data=[
                    [d, data["count"], data["total_xp"], data["total_gold"], data["avg_time_s"]]
                    for d, data in gameplay["quests_by_difficulty"].items()
                ],
            )
            run.log({"gameplay/quests_by_difficulty": diff_table})

    # Tool-calling comparison
    if tool_results:
        comparison_table = wandb.Table(
            columns=["model", "tool_call_valid_rate", "tool_selection_accuracy",
                     "argument_completeness", "has_tool_tags_rate"],
            data=[
                ["base", tool_results["base"]["tool_call_valid_rate"],
                 tool_results["base"]["tool_selection_accuracy"],
                 tool_results["base"]["argument_completeness_avg"],
                 tool_results["base"]["has_tool_tags_rate"]],
                ["fine-tuned", tool_results["fine_tuned"]["tool_call_valid_rate"],
                 tool_results["fine_tuned"]["tool_selection_accuracy"],
                 tool_results["fine_tuned"]["argument_completeness_avg"],
                 tool_results["fine_tuned"]["has_tool_tags_rate"]],
            ],
        )
        run.log({"evaluation/tool_calling_comparison": comparison_table})

        for key in ["tool_call_valid_rate", "tool_selection_accuracy",
                     "argument_completeness_avg", "has_tool_tags_rate"]:
            run.summary[f"base/{key}"] = tool_results["base"][key]
            run.summary[f"fine_tuned/{key}"] = tool_results["fine_tuned"][key]
            run.summary[f"delta/{key}"] = tool_results["fine_tuned"][key] - tool_results["base"][key]

    run.finish()
    print("\nEvaluation logged to W&B.")


if __name__ == "__main__":
    asyncio.run(main())
