"""
WoG MMORPG Agent — Hermes-2-Pro-Mistral-7B with MCP tool calling.
NVIDIA/CUDA version — uses transformers + bitsandbytes instead of MLX.

24/7 autonomous gaming agent that:
  1. Generates an Ethereum wallet locally
  2. Authenticates with the WoG shard via wallet signature
  3. Connects to the MCP server and discovers game tools
  4. Runs an autonomous game loop — fighting, questing, crafting, exploring
"""

import asyncio
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eth_account import Account
from eth_account.messages import encode_defunct
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import httpx

import weave

from memory import (
    load_memory, save_memory, memory_to_prompt,
    process_tool_result, handle_remember_command,
)
import wandb_logger
from policy_nvidia import PolicyEvaluator, get_current_strategy
from trajectory_logger import TrajectoryLogger

MODEL_ID = "NousResearch/Hermes-2-Pro-Mistral-7B"
MCP_URL = "https://mcp.urbantech.dev/mcp"
SHARD_URL = "https://wog.urbantech.dev"

# Per-agent identity — set AGENT_ID env var to differentiate parallel runs
AGENT_ID = os.environ.get("AGENT_ID", "0")

# Persist wallet key across restarts
WALLET_FILE = os.path.join(os.path.dirname(__file__), f".wallet_key_{AGENT_ID}")

# How often the agent acts (seconds) when running autonomously
TICK_INTERVAL = 3
# Max conversation history to keep (prevents context overflow for 7B model)
MAX_HISTORY = 20
# Max characters for a tool response fed back to the model
MAX_RESPONSE_CHARS = 1500
# How many consecutive no-tool-call cycles before resetting context
MAX_NO_TOOL_RETRIES = 3

SYSTEM_PROMPT_TEMPLATE = """\
You are a function calling AI model. You are an autonomous quest-chaining agent playing WoG, a fantasy MMORPG.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with gameplay. Always call a function.
Don't make assumptions about what values to plug into functions.

Here are the available tools:
<tools>
{tools}
</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "function_name", "arguments": {{"key": "value"}}}}
</tool_call>

Your SOLE focus is quests. Follow this loop every cycle:
1. Call get_my_status to check HP, active quests, level, and equipped armor/weapons.
2. If HP < 40%, heal or rest before anything else.
3. If you have an active quest, work toward completing its objective (fight, gather, navigate, talk to NPC).
4. If you have no active quest, call quests_get_catalog to find available quests, then quests_accept the highest-reward one.
5. When a quest objective is done, call quests_complete immediately to collect rewards.
6. After every 2-3 quest completions, do ALL of the following in order:
   a. Call technique_list_catalog and technique_learn every affordable skill you don't already have.
   b. Call equipment_get to check your current gear. If any armor slot (head, chest, legs, hands, feet) is empty, call shop_buy_item or craft_item to fill it immediately.
   c. Call shop_buy_item to upgrade any armor or weapon that is weaker than what is available.
   d. Call equipment_equip to equip everything you just bought or crafted.
7. Use travel_to_zone or navigate_to_npc to reach quest NPCs and objectives — never stand idle.

ARMOR RULE: Never fight with an empty armor slot. If you cannot afford to buy armor, call craft_item to craft it from materials. Equipped armor reduces deaths, which is critical.

{strategy}

{memory}"""


def load_or_create_wallet() -> Account:
    """Load existing wallet from disk or create a new one."""
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE) as f:
            key = f.read().strip()
        acct = Account.from_key(key)
        print(f"Loaded existing wallet: {acct.address}")
        return acct

    acct = Account.create()
    with open(WALLET_FILE, "w") as f:
        f.write(acct.key.hex())
    os.chmod(WALLET_FILE, 0o600)
    print(f"Created new wallet: {acct.address}")
    return acct


async def authenticate(wallet: Account) -> str:
    """Authenticate with the shard and return a JWT token."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{SHARD_URL}/auth/challenge", params={"wallet": wallet.address})
        r.raise_for_status()
        challenge = r.json()

        message = encode_defunct(text=challenge["message"])
        signed = wallet.sign_message(message)
        signature = signed.signature.hex()
        if not signature.startswith("0x"):
            signature = "0x" + signature

        r = await client.post(f"{SHARD_URL}/auth/verify", json={
            "walletAddress": wallet.address,
            "signature": signature,
            "timestamp": challenge["timestamp"],
        })
        r.raise_for_status()
        data = r.json()
        print(f"Authenticated! JWT expires in {data.get('expiresIn', '24h')}")
        return data["token"]


async def _lookup_character(client: httpx.AsyncClient, wallet_address: str, headers: dict) -> dict | None:
    """Return {entityId, zoneId, characterName} if wallet already has a character, else None."""
    try:
        r = await client.get(f"{SHARD_URL}/character/{wallet_address}", headers=headers)
        if r.status_code != 200:
            return None
        char_info = r.json()
        live = char_info.get("liveEntity")
        chars = char_info.get("characters", [])
        if live:
            return {
                "entityId": live.get("entityId", live.get("id", "")),
                "zoneId": live.get("zoneId", live.get("zone", "village-square")),
                "characterName": chars[0]["name"] if chars else "?",
            }
        if chars:
            return {
                "entityId": chars[0].get("tokenId", ""),
                "zoneId": "village-square",
                "characterName": chars[0]["name"],
            }
    except Exception as e:
        print(f"Character lookup error: {e}")
    return None


async def register_and_deploy(wallet: Account, token: str) -> dict:
    """Register wallet, create character if needed, and deploy agent.

    Fast path: if the wallet already has a character just deploy it directly,
    skipping the slow blockchain registration/minting steps that cause 524s.
    """
    timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # ── Fast path: existing character → skip registration ──────────────
        existing = await _lookup_character(client, wallet.address, headers)
        if existing:
            print(f"Existing character found: {existing['characterName']} — skipping registration")
        else:
            # ── Slow path: new wallet → register + mint (blockchain ops) ───
            print("New wallet — registering and creating character...")
            try:
                r = await client.post(f"{SHARD_URL}/wallet/register", json={
                    "walletAddress": wallet.address,
                })
                print(f"Wallet register: {r.status_code}")
            except (httpx.ReadTimeout, httpx.HTTPError) as e:
                print(f"Wallet register error (non-fatal): {e}")

            for attempt in range(5):
                try:
                    r = await client.post(f"{SHARD_URL}/character/create", json={
                        "walletAddress": wallet.address,
                        "name": f"HermesAgent{AGENT_ID}",
                        "race": "human",
                        "className": "warrior",
                    }, headers=headers)
                    if r.status_code == 200:
                        print(f"Character created: {r.json().get('character', {}).get('name', '?')}")
                    else:
                        print(f"Character create: {r.status_code} (may already exist)")
                    break
                except (httpx.ReadTimeout, httpx.HTTPError) as e:
                    wait = 2 ** attempt
                    print(f"character/create error ({attempt+1}/5): {e} — retrying in {wait}s...")
                    await asyncio.sleep(wait)

        # ── Deploy (try up to 3 times) ──────────────────────────────────────
        for attempt in range(3):
            try:
                r = await client.post(f"{SHARD_URL}/agent/deploy", json={
                    "walletAddress": wallet.address,
                }, headers=headers)
                if r.status_code == 200:
                    deploy = r.json()
                    print(f"Agent deployed: entity={deploy['entityId']} zone={deploy['zoneId']} "
                          f"char={deploy.get('characterName', '?')}")
                    return deploy
                print(f"Deploy attempt {attempt+1}: {r.status_code} — {r.text[:200]}")
            except (httpx.ReadTimeout, httpx.HTTPError) as e:
                print(f"Deploy attempt {attempt+1} error: {e}")
            await asyncio.sleep(3)

        # ── Final fallback: re-lookup character after deploy attempts ───────
        print("Falling back to character lookup...")
        info = await _lookup_character(client, wallet.address, headers)
        if info:
            print(f"Using character: {info['characterName']} entity={info['entityId']}")
            return info
        raise RuntimeError("Could not deploy or find character — check shard logs")


# Quest-chainer tool set — expose quest, skill, equipment, navigation tools
CORE_TOOL_PREFIXES = [
    "get_my_status",
    "quests_get_catalog", "quests_get_active", "quests_accept", "quests_complete",
    "technique_list_catalog", "technique_learn", "technique_cast",
    "shop_get_catalog", "shop_buy_item", "shop_get_item",
    "equipment_equip", "equipment_unequip", "equipment_get", "equipment_find_blacksmiths", "equipment_repair",
    "craft_item", "craft_list_recipes", "craft_get_recipe",
    "navigate_to_npc", "navigate_to", "travel_to_zone",
    "fight_until_dead", "grind_mobs",
    "scan_zone", "find_mobs_for_level",
    "items_get_inventory", "items_use",
    "heal", "rest",
]

# These params are auto-injected by the agent loop — hide from model
AUTO_INJECT_PARAMS = {"sessionId", "entityId", "zoneId"}


def format_tools_for_prompt(tools: list) -> str:
    """Format MCP tools as JSON function definitions (Hermes-2-Pro training format)."""
    tool_defs = []
    for t in tools:
        if not any(t.name.startswith(p) or t.name == p for p in CORE_TOOL_PREFIXES):
            continue
        props = {}
        if t.inputSchema and "properties" in t.inputSchema:
            for k, v in t.inputSchema["properties"].items():
                if k in AUTO_INJECT_PARAMS:
                    continue
                props[k] = v
        required = [r for r in (t.inputSchema.get("required", []) if t.inputSchema else [])
                     if r not in AUTO_INJECT_PARAMS]
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        tool_defs.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "")[:150],
                "parameters": schema,
            }
        })
    return json.dumps(tool_defs, separators=(",", ":"))


@weave.op()
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


def truncate_response(text: str) -> str:
    """Truncate tool responses to fit the 7B model's limited context."""
    if len(text) <= MAX_RESPONSE_CHARS:
        return text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            summary = {}
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 3:
                    summary[k] = v[:3]
                    summary[f"{k}_total"] = len(v)
                elif isinstance(v, dict) and "count" in v:
                    summary[k] = v
                else:
                    summary[k] = v
            truncated = json.dumps(summary, separators=(",", ":"))
            if len(truncated) <= MAX_RESPONSE_CHARS:
                return truncated
            return truncated[:MAX_RESPONSE_CHARS] + "...(truncated)"
    except (json.JSONDecodeError, TypeError):
        pass
    return text[:MAX_RESPONSE_CHARS] + "...(truncated)"


@weave.op()
def build_prompt(messages: list[dict]) -> str:
    """Build a ChatML prompt from a list of {role, content} messages."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep system prompt + last MAX_HISTORY messages to avoid context overflow."""
    if len(messages) <= MAX_HISTORY + 1:
        return messages
    return [messages[0]] + messages[-(MAX_HISTORY):]


# ── NVIDIA inference helpers ──

def load_model(model_id: str, adapter_path: str | None = None):
    """Load model + tokenizer using transformers with 8-bit quantization."""
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"LoRA adapter loaded from {adapter_path}")

    return model, tokenizer


@weave.op()
def run_inference(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Run inference on NVIDIA GPU via transformers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


async def main():
    # ── Step 1: Wallet ──
    wallet = load_or_create_wallet()

    # ── Step 2: Authenticate with shard ──
    print("Authenticating with WoG shard...")
    token = await authenticate(wallet)

    # ── Step 3: Register + deploy agent on shard ──
    print("Deploying agent...")
    deploy = await register_and_deploy(wallet, token)
    entity_id = deploy["entityId"]
    zone_id = deploy["zoneId"]

    # ── Step 4: Connect to MCP server ──
    print(f"\nConnecting to MCP server at {MCP_URL}...")
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as mcp:
            await mcp.initialize()

            session_id = get_session_id()
            print(f"MCP session: {session_id}")

            # Authenticate MCP session
            print("Authenticating MCP session...")
            challenge_result = await mcp.call_tool("auth_get_challenge", {"walletAddress": wallet.address})
            challenge_text = [c.text for c in challenge_result.content if hasattr(c, "text")][0]
            challenge_data = json.loads(challenge_text)

            message = encode_defunct(text=challenge_data["message"])
            signed = wallet.sign_message(message)
            sig_hex = "0x" + signed.signature.hex() if not signed.signature.hex().startswith("0x") else signed.signature.hex()

            auth_result = await mcp.call_tool("auth_verify_signature", {
                "walletAddress": wallet.address,
                "signature": sig_hex,
                "timestamp": challenge_data["timestamp"],
            })
            print(f"MCP auth: {[c.text for c in auth_result.content if hasattr(c, 'text')]}")

            # Discover entity
            print("Discovering entity via get_my_status...")
            for attempt in range(3):
                try:
                    status_result = await mcp.call_tool("get_my_status", {
                        "sessionId": session_id,
                        "entityId": entity_id,
                        "zoneId": zone_id,
                    })
                    status_text = [c.text for c in status_result.content if hasattr(c, "text")][0]
                    if "not authenticated" in status_text.lower():
                        print(f"  Attempt {attempt+1}: Not authenticated, re-authing...")
                        cr = await mcp.call_tool("auth_get_challenge", {"walletAddress": wallet.address})
                        ct = [c.text for c in cr.content if hasattr(c, "text")][0]
                        cd = json.loads(ct)
                        msg = encode_defunct(text=cd["message"])
                        s = wallet.sign_message(msg)
                        sh = "0x" + s.signature.hex() if not s.signature.hex().startswith("0x") else s.signature.hex()
                        await mcp.call_tool("auth_verify_signature", {
                            "walletAddress": wallet.address, "signature": sh, "timestamp": cd["timestamp"],
                        })
                        continue
                    status_data = json.loads(status_text)
                    if status_data.get("entityId"):
                        entity_id = status_data["entityId"]
                    if status_data.get("zone"):
                        zone_id = status_data["zone"]
                    print(f"Live status: entity={entity_id} zone={zone_id} "
                          f"L{status_data.get('level')} HP={status_data.get('hp')}/{status_data.get('maxHp')} "
                          f"gold={status_data.get('goldBalance')}")
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed: {e}")
            else:
                print("Could not get live status after 3 attempts — using deploy info")

            # Discover tools
            tools_result = await mcp.list_tools()
            tools = tools_result.tools
            print(f"Discovered {len(tools)} total tools.")

            tools_text = format_tools_for_prompt(tools)
            tool_defs = json.loads(tools_text)
            print(f"Showing {len(tool_defs)} core tools to model.\n")

            # ── Step 5: Load memory + model ──
            mem = load_memory()
            mem["stats"]["sessions"] = mem["stats"].get("sessions", 0) + 1
            save_memory(mem)

            policy_evaluator = PolicyEvaluator()

            def rebuild_system_prompt() -> str:
                return SYSTEM_PROMPT_TEMPLATE.format(
                    tools=tools_text,
                    strategy=get_current_strategy(mem),
                    memory=memory_to_prompt(mem),
                )

            print(f"Loading {MODEL_ID} (NVIDIA/CUDA)...")
            model, tokenizer = load_model(MODEL_ID)
            print(f"Model loaded on {model.device}.\n")

            wandb_logger.init_run({
                "model": MODEL_ID,
                "backend": "nvidia",
                "tick_interval": TICK_INTERVAL,
                "entity_id": entity_id,
                "zone_id": zone_id,
                "wallet": wallet.address,
                "session": mem["stats"].get("sessions", 0),
            })

            weave.init("wog-agent")
            traj_logger = TrajectoryLogger(agent_id=AGENT_ID)

            print("=" * 60)
            print("  WoG Agent is LIVE — NVIDIA/CUDA — running autonomously")
            print(f"  Wallet:  {wallet.address}")
            print(f"  Entity:  {entity_id}")
            print(f"  Zone:    {zone_id}")
            print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            print(f"  Memory:  {len(mem.get('journal', []))} journal entries, "
                  f"{len(mem.get('strategies', []))} strategies, "
                  f"session #{mem['stats']['sessions']}")
            print("=" * 60)
            print()

            # ── Step 6: Autonomous game loop ──
            messages: list[dict] = [{"role": "system", "content": rebuild_system_prompt()}]

            messages.append({
                "role": "user",
                "content": "You just spawned into the game. Check your status, then start playing autonomously. "
                           "Fight mobs, complete quests, gather resources, and get stronger. "
                           "Review your memory for context from previous sessions.",
            })

            cycle = 0
            consecutive_errors = 0
            consecutive_no_tool = 0
            MEMORY_REFRESH_INTERVAL = 10

            while True:
                cycle += 1
                messages = trim_history(messages)

                if cycle % MEMORY_REFRESH_INTERVAL == 0:
                    messages[0] = {"role": "system", "content": rebuild_system_prompt()}

                try:
                    prompt = build_prompt(messages)
                    stats_before = dict(mem.get("stats", {}))
                    traj_logger.begin_cycle(
                        cycle, messages, prompt, stats_before,
                        quests_completed=len(mem.get("quests", {}).get("completed", [])),
                        zones_discovered=len(mem.get("stats", {}).get("zone_visit_counts", {})),
                        zone=mem.get("facts", {}).get("zone"),
                        quest_completion_times=mem.get("stats", {}).get("quest_completion_times", []),
                    )

                    t0 = time.time()
                    response = run_inference(model, tokenizer, prompt)
                    inference_time = time.time() - t0

                    # Log GPU memory stats if CUDA is available
                    wandb_logger.log_gpu_stats(cycle)

                    tool_call = parse_tool_call(response)

                    if tool_call is None:
                        consecutive_no_tool += 1
                        print(f"[cycle {cycle}] No tool call ({consecutive_no_tool}/{MAX_NO_TOOL_RETRIES}): {response[:100]}")
                        wandb_logger.log_cycle(
                            cycle, None, None, response[:100], mem,
                            inference_time, len(prompt), False,
                        )

                        if consecutive_no_tool >= MAX_NO_TOOL_RETRIES:
                            print(f"[cycle {cycle}] Model stuck, resetting context...")
                            messages = [{"role": "system", "content": rebuild_system_prompt()}]
                            messages.append({
                                "role": "user",
                                "content": "Call get_my_status to check your situation.",
                            })
                            consecutive_no_tool = 0
                        else:
                            messages.append({"role": "assistant", "content": response})
                            messages.append({
                                "role": "user",
                                "content": "You MUST respond with a tool call:\n"
                                           "<tool_call>\n"
                                           '{"name": "get_my_status", "arguments": {}}\n'
                                           "</tool_call>",
                            })
                        await asyncio.sleep(TICK_INTERVAL)
                        continue

                    consecutive_no_tool = 0

                    name = tool_call.get("name", "")
                    args = tool_call.get("arguments", {})
                    print(f"[cycle {cycle}] {name}({json.dumps(args, separators=(',', ':'))})")

                    # Handle local "remember" pseudo-tool
                    if name == "remember":
                        result_text = handle_remember_command(mem, args)
                        print(f"  [memory] {result_text}")
                        messages.append({"role": "assistant", "content": response})
                        messages.append({
                            "role": "user",
                            "content": f"<tool_response>\n{result_text}\n</tool_response>\n\n"
                                       "Memory saved. Continue playing — call a game tool.",
                        })
                        await asyncio.sleep(TICK_INTERVAL)
                        continue

                    # Auto-inject session params
                    args.setdefault("sessionId", session_id)
                    args.setdefault("entityId", entity_id)
                    args.setdefault("zoneId", zone_id)

                    # Execute MCP tool call
                    try:
                        result = await mcp.call_tool(name, args)
                        result_text = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        consecutive_errors = 0
                    except Exception as e:
                        result_text = f"Error: {e}"
                        consecutive_errors += 1

                    preview = result_text[:300].replace("\n", " ")
                    print(f"  -> {preview}")

                    process_tool_result(mem, name, result_text)

                    # Log trajectory for fine-tuning
                    _comp_times = mem.get("quests", {}).get("completion_times", [])
                    _last_diff = _comp_times[-1].get("difficulty") if _comp_times else None

                    traj_logger.end_cycle(
                        response=response,
                        tool_call=tool_call,
                        tool_name=name,
                        tool_args=args,
                        tool_result=result_text,
                        tool_success=consecutive_errors == 0,
                        stats_after=dict(mem.get("stats", {})),
                        inference_time=inference_time,
                        quests_completed=len(mem.get("quests", {}).get("completed", [])),
                        zones_discovered=len(mem.get("stats", {}).get("zone_visit_counts", {})),
                        zone=mem.get("facts", {}).get("zone"),
                        quest_completion_times=mem.get("stats", {}).get("quest_completion_times", []),
                        last_quest_difficulty=_last_diff,
                    )

                    # Sync zone_id
                    new_zone = mem.get("facts", {}).get("zone")
                    if new_zone and new_zone != zone_id:
                        zone_id = new_zone

                    truncated = truncate_response(result_text)

                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{truncated}\n</tool_response>\n\n"
                                   "Decide your next action. Always call a tool.",
                    })

                    had_error = consecutive_errors > 0
                    wandb_logger.log_cycle(
                        cycle, name, args, preview, mem,
                        inference_time, len(prompt), had_error,
                    )

                    old_strategy = get_current_strategy(mem)
                    new_strategy = policy_evaluator.maybe_update(
                        cycle, mem, model, tokenizer,
                        tool_counts=dict(wandb_logger._tool_counts),
                    )
                    if new_strategy:
                        wandb_logger.log_policy_update(
                            cycle,
                            old_strategy,
                            new_strategy,
                            getattr(policy_evaluator, "last_deltas", {}),
                            getattr(policy_evaluator, "last_improvement_score", 0.0),
                            ema_score=getattr(policy_evaluator, "last_ema_score", 0.0),
                            action=getattr(policy_evaluator, "last_action", "adopted"),
                        )
                        save_memory(mem)
                        messages[0] = {"role": "system", "content": rebuild_system_prompt()}

                    if consecutive_errors >= 5:
                        print(f"[cycle {cycle}] Too many errors, cooling down 30s...")
                        await asyncio.sleep(30)
                        consecutive_errors = 0

                    await asyncio.sleep(TICK_INTERVAL)

                except KeyboardInterrupt:
                    print("\nShutting down agent...")
                    save_memory(mem)
                    print(f"Memory saved ({len(mem['journal'])} journal entries).")
                    traj_logger.close()
                    wandb_logger.finish(mem)
                    try:
                        await mcp.call_tool("character_logout", {
                            "sessionId": session_id,
                            "entityId": entity_id,
                            "zoneId": zone_id,
                        })
                        print("Character saved and logged out.")
                    except Exception:
                        pass
                    break

                except Exception as e:
                    print(f"[cycle {cycle}] Unexpected error: {e}")
                    wandb_logger.log_error(cycle, type(e).__name__, str(e))
                    save_memory(mem)
                    await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
