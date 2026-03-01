"""
WoG MMORPG Agent — Hermes-2-Pro-Mistral-7B with MCP tool calling.

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
import signal
import time

from eth_account import Account
from eth_account.messages import encode_defunct
from mlx_lm import load, generate
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import httpx

import weave

from memory import (
    load_memory, save_memory, memory_to_prompt,
    process_tool_result, handle_remember_command,
)
import wandb_logger
from policy import PolicyEvaluator, get_current_strategy
from trajectory_logger import TrajectoryLogger

MODEL_ID = "mlx-community/Hermes-2-Pro-Mistral-7B-8bit"
MCP_URL = "https://ips-latter-step-socket.trycloudflare.com/mcp"
SHARD_URL = "https://private-accessible-asia-qualified.trycloudflare.com"

# Persist wallet key across restarts
WALLET_FILE = os.path.join(os.path.dirname(__file__), ".wallet_key")

# How often the agent acts (seconds) when running autonomously
TICK_INTERVAL = 3
# Max conversation history to keep (prevents context overflow for 7B model)
MAX_HISTORY = 20
# Max characters for a tool response fed back to the model
MAX_RESPONSE_CHARS = 1500
# How many consecutive no-tool-call cycles before resetting context
MAX_NO_TOOL_RETRIES = 3

SYSTEM_PROMPT_TEMPLATE = """\
You are a function calling AI model. You are an autonomous agent playing WoG, a fantasy MMORPG.
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
        # Step 1: Get challenge
        r = await client.get(f"{SHARD_URL}/auth/challenge", params={"wallet": wallet.address})
        r.raise_for_status()
        challenge = r.json()

        # Step 2: Sign the challenge message
        message = encode_defunct(text=challenge["message"])
        signed = wallet.sign_message(message)
        signature = signed.signature.hex()
        if not signature.startswith("0x"):
            signature = "0x" + signature

        # Step 3: Verify signature and get JWT
        r = await client.post(f"{SHARD_URL}/auth/verify", json={
            "walletAddress": wallet.address,
            "signature": signature,
            "timestamp": challenge["timestamp"],
        })
        r.raise_for_status()
        data = r.json()
        print(f"Authenticated! JWT expires in {data.get('expiresIn', '24h')}")
        return data["token"]


async def register_and_deploy(wallet: Account, token: str) -> dict:
    """Register wallet, create character if needed, and deploy agent."""
    async with httpx.AsyncClient(timeout=120) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # Register wallet (idempotent — gives sFUEL + welcome gold)
        r = await client.post(f"{SHARD_URL}/wallet/register", json={
            "walletAddress": wallet.address,
        })
        if r.status_code == 200:
            print(f"Wallet registered: {r.json()}")
        else:
            print(f"Wallet register: {r.status_code} — {r.text}")

        # Create character if one doesn't exist yet
        r = await client.post(f"{SHARD_URL}/character/create", json={
            "walletAddress": wallet.address,
            "name": "HermesAgent",
            "race": "human",
            "className": "warrior",
        }, headers=headers)
        if r.status_code == 200:
            char = r.json()
            print(f"Character created: {char.get('character', {}).get('name', '?')}")
        else:
            print(f"Character create: {r.status_code} — {r.text[:200]} (may already exist)")

        # Deploy agent (spawns into the world)
        r = await client.post(f"{SHARD_URL}/agent/deploy", json={
            "walletAddress": wallet.address,
        }, headers=headers)
        if r.status_code == 200:
            deploy = r.json()
            print(f"Agent deployed: entity={deploy['entityId']} zone={deploy['zoneId']} "
                  f"char={deploy.get('characterName', '?')}")
            return deploy

        # Already live — look up existing character info
        print(f"Deploy: {r.status_code} — {r.text[:200]}")
        print("Looking up existing character...")
        r = await client.get(f"{SHARD_URL}/character/{wallet.address}", headers=headers)
        r.raise_for_status()
        char_info = r.json()
        # Use the first character's tokenId as a fallback entityId
        chars = char_info.get("characters", [])
        live = char_info.get("liveEntity")
        if live:
            entity_id = live.get("entityId", live.get("id", ""))
            zone_id = live.get("zoneId", live.get("zone", "village-square"))
        elif chars:
            entity_id = chars[0].get("tokenId", "")
            zone_id = "village-square"
        else:
            raise RuntimeError("No character found and deploy failed")
        name = chars[0]["name"] if chars else "?"
        print(f"Found existing character: {name} entity={entity_id} zone={zone_id}")
        return {"entityId": entity_id, "zoneId": zone_id, "characterName": name}


# Core tools a 7B model can handle — skip auth/admin/niche tools
CORE_TOOL_PREFIXES = [
    "get_my_status", "scan_zone", "fight_", "grind_mobs",
    "gather_", "craft_", "quest", "shop_", "equip_", "unequip_",
    "use_item", "inventory", "heal", "rest", "travel", "transition",
    "move_", "look_", "check_", "buy_", "sell_",
]

# These params are auto-injected by the agent loop — hide from model
AUTO_INJECT_PARAMS = {"sessionId", "entityId", "zoneId"}


def format_tools_for_prompt(tools: list) -> str:
    """Format MCP tools as JSON function definitions (Hermes-2-Pro training format).
    Filters to core gameplay tools and hides auto-injected params."""
    tool_defs = []
    for t in tools:
        # Filter to core tools
        if not any(t.name.startswith(p) or t.name == p for p in CORE_TOOL_PREFIXES):
            continue
        # Build filtered schema (hide auto-injected params)
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
    """Extract a tool call JSON block from model output.
    Tries <tool_call> tags first, then falls back to bare JSON with 'name' key."""
    # Try tagged format first
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: look for any JSON object with a "name" key
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
            if "name" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def truncate_response(text: str) -> str:
    """Truncate tool responses to fit the 7B model's limited context.
    Tries to parse JSON and summarize, otherwise hard-truncates."""
    if len(text) <= MAX_RESPONSE_CHARS:
        return text
    try:
        data = json.loads(text)
        # For scan_zone: summarize mobs/resources instead of full dump
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
    # Always keep the system prompt (index 0)
    return [messages[0]] + messages[-(MAX_HISTORY):]


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

            # Get the MCP session ID — this is what game tools need as sessionId
            session_id = get_session_id()
            print(f"MCP session: {session_id}")

            # Authenticate MCP session using MCP's own challenge
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

            # Discover real entity ID via get_my_status
            # (the shard deploy may return a token ID, not the live entity UUID)
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
                        # Re-authenticate
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

            print(f"Loading {MODEL_ID}...")
            model, tokenizer = load(MODEL_ID)
            print("Model loaded.\n")

            wandb_logger.init_run({
                "model": MODEL_ID,
                "tick_interval": TICK_INTERVAL,
                "entity_id": entity_id,
                "zone_id": zone_id,
                "wallet": wallet.address,
                "session": mem["stats"].get("sessions", 0),
            })

            # ── Weave tracing + trajectory logging ──
            weave.init("wog-agent")
            traj_logger = TrajectoryLogger()

            print("=" * 60)
            print("  WoG Agent is LIVE — running autonomously 24/7")
            print(f"  Wallet:  {wallet.address}")
            print(f"  Entity:  {entity_id}")
            print(f"  Zone:    {zone_id}")
            print(f"  Memory:  {len(mem.get('journal', []))} journal entries, "
                  f"{len(mem.get('strategies', []))} strategies, "
                  f"session #{mem['stats']['sessions']}")
            print("=" * 60)
            print()

            # ── Shutdown helper — always logout, no matter how we exit ──
            async def shutdown(reason: str = "unknown"):
                """Logout character, save memory, close loggers. Safe to call multiple times."""
                if getattr(shutdown, "_done", False):
                    return
                shutdown._done = True
                print(f"\nShutting down agent ({reason})...")
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
                    print("Character logged out from shard.")
                except Exception as e:
                    print(f"Logout failed: {e} — trying HTTP fallback...")
                    # Fallback: hit the shard REST API directly
                    try:
                        async with httpx.AsyncClient(timeout=10) as hc:
                            await hc.post(f"{SHARD_URL}/agent/undeploy", json={
                                "walletAddress": wallet.address,
                            }, headers={"Authorization": f"Bearer {token}"})
                        print("Agent undeployed via HTTP fallback.")
                    except Exception:
                        print("HTTP fallback also failed — agent may remain on shard until timeout.")

            # Wire up OS signals so kill/SIGTERM also triggers logout
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(shutdown(f"signal {s.name}")))

            # ── Step 6: Autonomous game loop ──
            messages: list[dict] = [{"role": "system", "content": rebuild_system_prompt()}]

            # Kick off with an initial instruction
            messages.append({
                "role": "user",
                "content": "You just spawned into the game. Check your status, then start playing autonomously. "
                           "Fight mobs, complete quests, gather resources, and get stronger. "
                           "Review your memory for context from previous sessions.",
            })

            cycle = 0
            consecutive_errors = 0
            consecutive_no_tool = 0
            # Refresh system prompt with latest memory every N cycles
            MEMORY_REFRESH_INTERVAL = 10

            try:
                while True:
                    cycle += 1
                    messages = trim_history(messages)

                    # Periodically refresh system prompt with latest memory
                    if cycle % MEMORY_REFRESH_INTERVAL == 0:
                        messages[0] = {"role": "system", "content": rebuild_system_prompt()}

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
                    response = generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=512,
                        verbose=False,
                    )
                    inference_time = time.time() - t0

                    tool_call = parse_tool_call(response)

                    if tool_call is None:
                        consecutive_no_tool += 1
                        print(f"[cycle {cycle}] No tool call ({consecutive_no_tool}/{MAX_NO_TOOL_RETRIES}): {response[:100]}")
                        wandb_logger.log_cycle(
                            cycle, None, None, response[:100], mem,
                            inference_time, len(prompt), False,
                        )

                        if consecutive_no_tool >= MAX_NO_TOOL_RETRIES:
                            # Model is stuck — reset context completely
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

                    # Auto-inject session params the model doesn't need to know
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

                    # Print abbreviated result
                    preview = result_text[:300].replace("\n", " ")
                    print(f"  -> {preview}")

                    # Auto-extract memory from tool results (uses full text)
                    process_tool_result(mem, name, result_text)

                    # Log trajectory for fine-tuning
                    # Get difficulty of most recent quest completion (if any this cycle)
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

                    # Sync zone_id when agent travels so future tool calls use the right zone
                    new_zone = mem.get("facts", {}).get("zone")
                    if new_zone and new_zone != zone_id:
                        zone_id = new_zone

                    # Truncate before feeding back to model
                    truncated = truncate_response(result_text)

                    # Feed back into conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{truncated}\n</tool_response>\n\n"
                                   "Decide your next action. Always call a tool.",
                    })

                    # W&B telemetry
                    had_error = consecutive_errors > 0
                    wandb_logger.log_cycle(
                        cycle, name, args, preview, mem,
                        inference_time, len(prompt), had_error,
                    )

                    # Policy self-improvement check
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

                    # Back off if too many consecutive errors
                    if consecutive_errors >= 5:
                        print(f"[cycle {cycle}] Too many errors, cooling down 30s...")
                        await asyncio.sleep(30)
                        consecutive_errors = 0

                    await asyncio.sleep(TICK_INTERVAL)

            except KeyboardInterrupt:
                await shutdown("Ctrl+C")
            except Exception as e:
                print(f"[cycle {cycle}] Fatal error: {e}")
                wandb_logger.log_error(cycle, type(e).__name__, str(e))
                await shutdown(f"crash: {e}")
                raise
            finally:
                # Catch-all: if nothing above triggered shutdown yet, do it now
                await shutdown("process exit")


if __name__ == "__main__":
    asyncio.run(main())
