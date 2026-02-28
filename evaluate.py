"""
Evaluation script: base model vs LoRA fine-tuned model.

Uses Weave to run side-by-side evaluation with custom scorers
for tool call validity, selection accuracy, and argument completeness.

Usage:
    python evaluate.py [--adapter-path adapters] [--data data/valid.jsonl]
"""

import argparse
import json
import re
import asyncio
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


async def run_evaluation(agent: WoGAgent, dataset: list[dict], label: str) -> dict:
    """Run evaluation on dataset and collect scores."""
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
    parser.add_argument("--max-examples", type=int, default=50, help="Max examples to evaluate")
    args = parser.parse_args()

    weave.init("wog-agent")

    # Load dataset
    print(f"Loading evaluation data from {args.data}...")
    dataset = load_eval_dataset(args.data)
    if len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
    print(f"  Evaluating on {len(dataset)} examples")

    # Create both agents
    print(f"\nLoading base model: {args.base_model}")
    base_agent = WoGAgent(model_id=args.base_model, label="base")
    base_agent.load_model()

    print(f"Loading fine-tuned model: {args.base_model} + {args.adapter_path}")
    ft_agent = WoGAgent(
        model_id=args.base_model,
        adapter_path=args.adapter_path,
        label="fine-tuned",
    )
    ft_agent.load_model()

    # Run evaluations
    print("\nEvaluating base model...")
    base_results = await run_evaluation(base_agent, dataset, "base")

    print("\nEvaluating fine-tuned model...")
    ft_results = await run_evaluation(ft_agent, dataset, "fine-tuned")

    # Print comparison
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 62)

    metrics = [
        ("Tool Call Valid Rate", "tool_call_valid_rate"),
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("Argument Completeness", "argument_completeness_avg"),
        ("Has Tool Tags Rate", "has_tool_tags_rate"),
    ]

    for display_name, key in metrics:
        base_val = base_results[key]
        ft_val = ft_results[key]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{display_name:<30} {base_val:>10.3f} {ft_val:>12.3f} {sign}{delta:>9.3f}")

    print("=" * 60)

    # Log to W&B via weave
    import wandb
    run = wandb.init(project="wog-agent", job_type="evaluation")

    comparison_table = wandb.Table(
        columns=["model", "tool_call_valid_rate", "tool_selection_accuracy",
                 "argument_completeness", "has_tool_tags_rate"],
        data=[
            ["base", base_results["tool_call_valid_rate"],
             base_results["tool_selection_accuracy"],
             base_results["argument_completeness_avg"],
             base_results["has_tool_tags_rate"]],
            ["fine-tuned", ft_results["tool_call_valid_rate"],
             ft_results["tool_selection_accuracy"],
             ft_results["argument_completeness_avg"],
             ft_results["has_tool_tags_rate"]],
        ],
    )
    run.log({"evaluation/comparison": comparison_table})

    for key in ["tool_call_valid_rate", "tool_selection_accuracy",
                "argument_completeness_avg", "has_tool_tags_rate"]:
        run.summary[f"base/{key}"] = base_results[key]
        run.summary[f"fine_tuned/{key}"] = ft_results[key]
        run.summary[f"delta/{key}"] = ft_results[key] - base_results[key]

    run.finish()
    print("\nEvaluation logged to W&B.")


if __name__ == "__main__":
    asyncio.run(main())
