"""
Evaluation script: base model vs LoRA fine-tuned model.
NVIDIA/CUDA version — uses transformers + PEFT instead of mlx_lm.

Usage:
    python evaluate_nvidia.py [--adapter-path adapters] [--data data/valid.jsonl]
"""

import argparse
import json
import re
import asyncio

import torch
import weave
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


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
    """WoG agent model wrapper for Weave evaluation (NVIDIA)."""
    model_id: str = "NousResearch/Hermes-2-Pro-Mistral-7B"
    adapter_path: str | None = None
    label: str = "base"

    model_config = {"arbitrary_types_allowed": True}

    def _load(self):
        if hasattr(self, "_loaded") and self._loaded:
            return
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if self.adapter_path:
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            print(f"[{self.label}] LoRA adapter loaded from {self.adapter_path}")
        self._loaded = True

    @weave.op()
    def predict(self, system_prompt: str, user_message: str) -> dict:
        self._load()
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
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
    has_call = model_output.get("has_tool_call", False)
    tool_call = model_output.get("tool_call")
    valid = False
    if has_call and tool_call:
        valid = bool(tool_call.get("name")) and "arguments" in tool_call
    return {"tool_call_valid": valid}


@weave.op()
def tool_selection_accuracy(model_output: dict, expected_tool: str) -> dict:
    predicted = model_output.get("tool_name")
    correct = predicted == expected_tool if expected_tool else False
    return {"tool_selection_correct": correct}


@weave.op()
def argument_completeness(model_output: dict, expected_args: dict) -> dict:
    predicted_args = model_output.get("tool_args") or {}
    if not expected_args:
        return {"argument_completeness": 1.0}
    expected_keys = set(expected_args.keys()) - {"sessionId", "entityId", "zoneId"}
    if not expected_keys:
        return {"argument_completeness": 1.0}
    predicted_keys = set(predicted_args.keys()) - {"sessionId", "entityId", "zoneId"}
    overlap = expected_keys & predicted_keys
    return {"argument_completeness": len(overlap) / len(expected_keys)}


@weave.op()
def response_quality(model_output: dict) -> dict:
    response = model_output.get("response", "")
    return {
        "has_tool_tags": "<tool_call>" in response and "</tool_call>" in response,
        "response_length": len(response),
        "not_empty": len(response.strip()) > 0,
    }


def load_eval_dataset(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            system_msg = user_msg = assistant_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]
            expected_call = parse_tool_call(assistant_msg)
            examples.append({
                "system_prompt": system_msg,
                "user_message": user_msg,
                "expected_tool": expected_call.get("name") if expected_call else None,
                "expected_args": expected_call.get("arguments", {}) if expected_call else {},
            })
    return examples


async def run_evaluation(agent: WoGAgent, dataset: list[dict], label: str) -> dict:
    results = {
        "label": label, "total": len(dataset),
        "tool_call_valid": 0, "tool_selection_correct": 0,
        "argument_completeness_sum": 0.0, "has_tool_tags": 0,
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
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned WoG agent (NVIDIA)")
    parser.add_argument("--base-model", default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--data", default="data/valid.jsonl")
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    weave.init("wog-agent")

    print(f"Loading evaluation data from {args.data}...")
    dataset = load_eval_dataset(args.data)
    if len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
    print(f"  Evaluating on {len(dataset)} examples")

    print(f"\nLoading base model: {args.base_model}")
    base_agent = WoGAgent(model_id=args.base_model, label="base")
    base_agent._load()

    print(f"Loading fine-tuned model: {args.base_model} + {args.adapter_path}")
    ft_agent = WoGAgent(model_id=args.base_model, adapter_path=args.adapter_path, label="fine-tuned")
    ft_agent._load()

    print("\nEvaluating base model...")
    base_results = await run_evaluation(base_agent, dataset, "base")

    print("\nEvaluating fine-tuned model...")
    ft_results = await run_evaluation(ft_agent, dataset, "fine-tuned")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 62)
    for display_name, key in [
        ("Tool Call Valid Rate", "tool_call_valid_rate"),
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("Argument Completeness", "argument_completeness_avg"),
        ("Has Tool Tags Rate", "has_tool_tags_rate"),
    ]:
        base_val = base_results[key]
        ft_val = ft_results[key]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{display_name:<30} {base_val:>10.3f} {ft_val:>12.3f} {sign}{delta:>9.3f}")
    print("=" * 60)

    import wandb
    eval_run = wandb.init(project="wog-agent", job_type="evaluation")
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
    eval_run.log({"evaluation/comparison": comparison_table})
    for key in ["tool_call_valid_rate", "tool_selection_accuracy",
                "argument_completeness_avg", "has_tool_tags_rate"]:
        eval_run.summary[f"base/{key}"] = base_results[key]
        eval_run.summary[f"fine_tuned/{key}"] = ft_results[key]
        eval_run.summary[f"delta/{key}"] = ft_results[key] - base_results[key]
    eval_run.finish()
    print("\nEvaluation logged to W&B.")


if __name__ == "__main__":
    asyncio.run(main())
