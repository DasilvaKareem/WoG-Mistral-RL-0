"""
Live model comparison: Base vs SFT vs Policy adapter.

Runs each variant as a real WoG agent for --cycles cycles,
starting with --starting-gold copper each.
Logs side-by-side metrics to W&B under the same run.

Usage (launched by modal_runner.py::run_comparison):
    python evaluate_comparison.py --variant base --cycles 100 --starting-gold 500
    python evaluate_comparison.py --variant sft  --adapter-path adapters/sft --cycles 100 --starting-gold 500
    python evaluate_comparison.py --variant policy --adapter-path adapters/policy --cycles 100 --starting-gold 500
"""

import argparse
import asyncio
import json
import os
import re
import time

import torch
import wandb
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from eth_account import Account

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL = "NousResearch/Hermes-2-Pro-Mistral-7B"
MCP_URL    = os.environ.get("MCP_URL", "https://mcp.urbantech.dev/mcp")
SHARD_URL  = os.environ.get("SHARD_URL", "https://wog.urbantech.dev")
AGENT_ID   = int(os.environ.get("AGENT_ID", "0"))
TICK       = 2.0


# ── Wallet ────────────────────────────────────────────────────────────────────
def load_wallet(agent_id: int) -> Account:
    key_file = f".wallet_key_{agent_id}"
    if os.path.exists(key_file):
        return Account.from_key(open(key_file).read().strip())
    acct = Account.create()
    open(key_file, "w").write(acct.key.hex())
    return acct


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(adapter_path: str | None):
    print(f"Loading model (adapter={adapter_path or 'none'})...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded adapter from {adapter_path}")
    else:
        print("  Running base model (no adapter)")
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_tool_call(text: str) -> dict | None:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
            if "name" in obj:
                return obj
        except Exception:
            pass
    return None


# ── Auth helpers ──────────────────────────────────────────────────────────────
async def shard_auth(wallet: Account) -> tuple[str, str, str]:
    """Authenticate with WoG shard, return (jwt, entity_id, zone_id)."""
    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{SHARD_URL}/auth/challenge",
                              json={"walletAddress": wallet.address})
        challenge = r.json()["challenge"]
        msg = f"\x19Ethereum Signed Message:\n{len(challenge)}{challenge}"
        sig = Account.sign_message(
            encode_defunct(text=challenge), private_key=wallet.key
        ).signature.hex()
        r = await client.post(f"{SHARD_URL}/auth/verify",
                              json={"walletAddress": wallet.address, "signature": sig})
        data = r.json()
        return data["token"], data.get("entityId", ""), data.get("zoneId", "village-square")


SYSTEM_PROMPT = """\
You are a function calling AI agent playing WoG. Call tools to complete quests, buy weapons, learn skills, and earn gold. Always call exactly one tool per response.
<tools>{tools}</tools>
For each call: <tool_call>{{"name": "tool_name", "arguments": {{"key": "value"}}}}</tool_call>"""


def build_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── Main eval loop ────────────────────────────────────────────────────────────
async def run_eval(variant: str, adapter_path: str | None, cycles: int, starting_gold: int):
    wallet = load_wallet(AGENT_ID)
    model, tokenizer = load_model(adapter_path)

    # W&B init
    run = wandb.init(
        project="wog-agent",
        job_type="comparison",
        group="model_comparison",
        name=f"{variant}_agent{AGENT_ID}",
        config={
            "variant": variant,
            "adapter": adapter_path,
            "cycles": cycles,
            "starting_gold": starting_gold,
            "agent_id": AGENT_ID,
        },
        reinit=True,
    )
    print(f"[wandb] {run.url}")

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()

            # Discover tools
            tools_result = await mcp.list_tools()
            tools_json = json.dumps([
                {"name": t.name, "desc": (t.description or "")[:80]}
                for t in tools_result.tools
            ])

            # Get session + entity info from get_my_status
            status_result = await mcp.call_tool("get_my_status", {
                "sessionId": "", "entityId": "", "zoneId": "village-square",
                "walletAddress": wallet.address,
            })
            status_text = "\n".join(c.text for c in status_result.content if hasattr(c, "text"))
            try:
                status_data = json.loads(status_text)
                entity_id = status_data.get("entityId", "")
                zone_id   = status_data.get("zone", "village-square")
                session_id = ""
            except Exception:
                entity_id = zone_id = session_id = ""

            # Give starting gold via buy/sell cycle if possible
            if starting_gold > 0:
                print(f"[{variant}] Attempting to set starting gold={starting_gold}...")
                try:
                    await mcp.call_tool("debug_set_gold", {
                        "sessionId": session_id, "entityId": entity_id,
                        "zoneId": zone_id, "walletAddress": wallet.address,
                        "amount": starting_gold,
                    })
                    print(f"[{variant}] Starting gold set to {starting_gold}")
                except Exception:
                    print(f"[{variant}] No debug_set_gold tool — agents start with existing gold")

            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_json)}]

            # Metrics
            quests_completed = 0
            total_gold = 0
            total_xp = 0
            total_kills = 0
            total_deaths = 0
            tool_errors = 0

            for cycle in range(1, cycles + 1):
                prompt = build_prompt(messages)
                t0 = time.time()
                response = run_inference(model, tokenizer, prompt)
                inference_time = time.time() - t0

                tool_call = parse_tool_call(response)
                result_text = ""
                tool_name = None
                tool_success = False

                if tool_call:
                    tool_name = tool_call.get("name", "")
                    args = tool_call.get("arguments", {})
                    args.setdefault("sessionId", session_id)
                    args.setdefault("entityId", entity_id)
                    args.setdefault("zoneId", zone_id)
                    args.setdefault("walletAddress", wallet.address)
                    try:
                        result = await mcp.call_tool(tool_name, args)
                        result_text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                        tool_success = True

                        # Parse metrics from status responses
                        try:
                            data = json.loads(result_text)
                            if "totalGold" in data:
                                total_gold = int(data.get("totalGold", total_gold))
                            if "totalXp" in data or "xp" in data:
                                total_xp = int(data.get("totalXp", data.get("xp", total_xp)))
                            if "zone" in data:
                                zone_id = data["zone"]
                        except Exception:
                            pass

                        if "quest" in tool_name and "complete" in tool_name.lower():
                            quests_completed += 1
                        if "killed" in result_text.lower() or "defeated" in result_text.lower():
                            total_kills += 1
                        if "died" in result_text.lower() or "death" in result_text.lower():
                            total_deaths += 1

                    except Exception as e:
                        result_text = f"Error: {e}"
                        tool_errors += 1
                else:
                    tool_errors += 1

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"<tool_response>\n{result_text[:400]}\n</tool_response>"})
                if len(messages) > 22:
                    messages = [messages[0]] + messages[-20:]

                # Log to W&B every cycle
                wandb.log({
                    f"{variant}/quests_completed": quests_completed,
                    f"{variant}/total_gold": total_gold,
                    f"{variant}/total_xp": total_xp,
                    f"{variant}/total_kills": total_kills,
                    f"{variant}/total_deaths": total_deaths,
                    f"{variant}/tool_errors": tool_errors,
                    f"{variant}/inference_time_s": inference_time,
                    f"{variant}/tool_success_rate": (cycle - tool_errors) / cycle * 100,
                    f"{variant}/tool_name": tool_name or "none",
                    "cycle": cycle,
                }, step=cycle)

                if cycle % 10 == 0:
                    print(f"[{variant}] cycle={cycle}/{cycles} quests={quests_completed} gold={total_gold} xp={total_xp} errors={tool_errors}")

                await asyncio.sleep(TICK)

    # Final summary
    wandb.run.summary.update({
        f"{variant}/final_quests": quests_completed,
        f"{variant}/final_gold": total_gold,
        f"{variant}/final_xp": total_xp,
        f"{variant}/final_kills": total_kills,
        f"{variant}/final_deaths": total_deaths,
        f"{variant}/total_tool_errors": tool_errors,
        f"{variant}/tool_success_rate_pct": (cycles - tool_errors) / cycles * 100,
    })
    run.finish()
    print(f"\n[{variant}] DONE — quests={quests_completed} gold={total_gold} xp={total_xp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["base", "sft", "policy"])
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--starting-gold", type=int, default=500)
    args = parser.parse_args()

    asyncio.run(run_eval(args.variant, args.adapter_path, args.cycles, args.starting_gold))


if __name__ == "__main__":
    from eth_account.messages import encode_defunct
    main()
