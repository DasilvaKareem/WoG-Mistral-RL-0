"""
Persistent memory system for the WoG agent.

Stores structured knowledge in a JSON file that survives restarts
and gets injected into the system prompt every cycle. The agent can
write memories via a local pseudo-tool (not an MCP tool).

Memory sections:
  - facts:       Key things about the character (level, class, gear, etc.)
  - quests:      Quest progress and completion history
  - zones:       Zones explored, what's in each one
  - strategies:  What works and what doesn't (mob priorities, routes, etc.)
  - inventory:   Notable items, crafting materials
  - journal:     Recent event log (auto-trimmed to last 30 entries)
"""

import json
import os
from datetime import datetime

MEMORY_FILE = os.path.join(os.path.dirname(__file__), ".memory.json")
MAX_JOURNAL = 30
MAX_FACTS = 20
MAX_STRATEGIES = 15
MAX_ZONES = 15

# Track last-seen gold for delta calculation
_last_gold: int | None = None


def _default_memory() -> dict:
    return {
        "facts": {},
        "quests": {"completed": [], "active": []},
        "zones": {},
        "strategies": [],
        "inventory_notes": [],
        "journal": [],
        "stats": {
            "total_kills": 0,
            "total_deaths": 0,
            "total_xp": 0,
            "total_gold_earned": 0,
            "sessions": 0,
            "first_seen": datetime.now().isoformat(),
        },
        "policy_history": [],
    }


def load_memory() -> dict:
    """Load memory from disk, or create default."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return _default_memory()


def save_memory(mem: dict) -> None:
    """Persist memory to disk."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)


def memory_to_prompt(mem: dict) -> str:
    """Format memory as a concise block for the system prompt."""
    lines = ["## Your Memory (persistent across restarts)"]

    # Facts
    if mem.get("facts"):
        lines.append("\n### Facts")
        for k, v in list(mem["facts"].items())[:MAX_FACTS]:
            lines.append(f"- {k}: {v}")

    # Quests
    quests = mem.get("quests", {})
    if quests.get("active"):
        lines.append("\n### Active Quests")
        for q in quests["active"][-5:]:
            lines.append(f"- {q}")
    if quests.get("completed"):
        lines.append(f"\n### Completed Quests ({len(quests['completed'])} total)")
        for q in quests["completed"][-5:]:
            lines.append(f"- {q}")

    # Zones
    if mem.get("zones"):
        lines.append("\n### Zones Explored")
        for zone_id, info in list(mem["zones"].items())[:MAX_ZONES]:
            lines.append(f"- {zone_id}: {info}")

    # Strategies
    if mem.get("strategies"):
        lines.append("\n### Strategies (things you've learned)")
        for s in mem["strategies"][-MAX_STRATEGIES:]:
            lines.append(f"- {s}")

    # Inventory notes
    if mem.get("inventory_notes"):
        lines.append("\n### Inventory Notes")
        for n in mem["inventory_notes"][-10:]:
            lines.append(f"- {n}")

    # Recent journal
    if mem.get("journal"):
        lines.append(f"\n### Recent Journal (last {min(len(mem['journal']), 10)})")
        for entry in mem["journal"][-10:]:
            lines.append(f"- {entry}")

    # Lifetime stats
    stats = mem.get("stats", {})
    if stats:
        lines.append(f"\n### Lifetime Stats")
        lines.append(f"- Kills: {stats.get('total_kills', 0)} | "
                     f"Deaths: {stats.get('total_deaths', 0)} | "
                     f"XP: {stats.get('total_xp', 0)} | "
                     f"Gold earned: {stats.get('total_gold_earned', 0)} | "
                     f"Sessions: {stats.get('sessions', 0)}")

    return "\n".join(lines)


# ── Auto-extraction from tool results ──

def extract_from_status(mem: dict, result: dict) -> None:
    """Update memory facts from a get_my_status result."""
    global _last_gold
    if not isinstance(result, dict):
        return
    mem["facts"]["level"] = result.get("level")
    mem["facts"]["class"] = result.get("class")
    mem["facts"]["race"] = result.get("race")
    mem["facts"]["zone"] = result.get("zone")
    mem["facts"]["hp"] = f"{result.get('hpPct', '?')}%"
    mem["facts"]["gold"] = result.get("goldBalance", "?")
    mem["facts"]["xp_progress"] = f"{result.get('xpPct', '?')}% to next level"

    # Track gold deltas
    current_gold = result.get("goldBalance")
    if current_gold is not None and isinstance(current_gold, (int, float)):
        if _last_gold is not None:
            delta = current_gold - _last_gold
            if delta > 0:
                mem["stats"]["total_gold_earned"] = mem["stats"].get("total_gold_earned", 0) + delta
        _last_gold = int(current_gold)

    if result.get("isDead"):
        mem["facts"]["status"] = "DEAD — need to recover"
    elif result.get("hpPct", 100) < 30:
        mem["facts"]["status"] = "LOW HP — need healing"
    else:
        mem["facts"]["status"] = "healthy"
    if result.get("gearWarnings"):
        mem["facts"]["gear_warnings"] = ", ".join(result["gearWarnings"])
    save_memory(mem)


def extract_from_grind(mem: dict, result: dict) -> None:
    """Update memory from a grind_mobs result."""
    if not isinstance(result, dict):
        return
    summary = result.get("summary", {})
    kills = summary.get("mobsKilled", 0)
    xp = summary.get("totalXpGained", 0)
    deaths = summary.get("playerDeaths", 0)
    gold = summary.get("goldEarned", summary.get("goldGained", 0))

    mem["stats"]["total_kills"] = mem["stats"].get("total_kills", 0) + kills
    mem["stats"]["total_xp"] = mem["stats"].get("total_xp", 0) + xp
    mem["stats"]["total_deaths"] = mem["stats"].get("total_deaths", 0) + deaths
    if gold and isinstance(gold, (int, float)) and gold > 0:
        mem["stats"]["total_gold_earned"] = mem["stats"].get("total_gold_earned", 0) + gold

    log = result.get("log", [])
    timestamp = datetime.now().strftime("%H:%M")
    if kills > 0:
        journal_entry = f"[{timestamp}] Grind: killed {kills} mobs, +{xp} XP, +{gold} gold"
        if deaths > 0:
            journal_entry += f", died {deaths}x"
        mem["journal"].append(journal_entry)

    # Check for level ups in log
    for entry in log:
        if "LEVEL UP" in entry:
            mem["journal"].append(f"[{timestamp}] {entry}")

    mem["journal"] = mem["journal"][-MAX_JOURNAL:]
    save_memory(mem)


def extract_from_fight(mem: dict, result: dict) -> None:
    """Update memory from a fight_until_dead result."""
    if not isinstance(result, dict):
        return
    timestamp = datetime.now().strftime("%H:%M")

    if result.get("killed"):
        target = result.get("target", {})
        xp = result.get("xpGained", 0)
        gold = result.get("goldEarned", result.get("goldGained", 0))
        mem["stats"]["total_kills"] = mem["stats"].get("total_kills", 0) + 1
        mem["stats"]["total_xp"] = mem["stats"].get("total_xp", 0) + xp
        if gold and isinstance(gold, (int, float)) and gold > 0:
            mem["stats"]["total_gold_earned"] = mem["stats"].get("total_gold_earned", 0) + gold
        mem["journal"].append(
            f"[{timestamp}] Killed {target.get('name', '?')} (L{target.get('level', '?')}), +{xp} XP, +{gold} gold"
        )
        if result.get("leveledUp"):
            mem["journal"].append(f"[{timestamp}] LEVEL UP to {result.get('newLevel')}!")
            mem["facts"]["level"] = result.get("newLevel")

    if result.get("playerDied"):
        mem["stats"]["total_deaths"] = mem["stats"].get("total_deaths", 0) + 1
        mem["journal"].append(f"[{timestamp}] DIED fighting {result.get('target', {}).get('name', '?')}")

    mem["journal"] = mem["journal"][-MAX_JOURNAL:]
    save_memory(mem)


def extract_from_scan(mem: dict, result: dict) -> None:
    """Update memory zone info from scan_zone result."""
    if not isinstance(result, dict):
        return
    zone_id = result.get("zone")
    if not zone_id:
        return
    mob_count = result.get("mobs", {}).get("count", 0)
    ore_count = result.get("oreNodes", {}).get("count", 0)
    flower_count = result.get("flowerNodes", {}).get("count", 0)
    players = len(result.get("otherPlayers", []))
    mem["zones"][zone_id] = (
        f"{mob_count} mobs, {ore_count} ore, {flower_count} flowers, {players} players"
    )
    save_memory(mem)


def extract_from_quest(mem: dict, tool_name: str, result_text: str) -> None:
    """Update memory from quest tool results."""
    try:
        result = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return

    timestamp = datetime.now().strftime("%H:%M")

    if tool_name == "quests_accept" and isinstance(result, dict):
        quest_name = result.get("questId") or result.get("name") or "unknown quest"
        mem["quests"]["active"].append(quest_name)
        mem["journal"].append(f"[{timestamp}] Accepted quest: {quest_name}")

    elif tool_name == "quests_complete" and isinstance(result, dict):
        quest_name = result.get("questId") or result.get("name") or "unknown quest"
        if quest_name in mem["quests"]["active"]:
            mem["quests"]["active"].remove(quest_name)
        if quest_name not in mem["quests"]["completed"]:
            mem["quests"]["completed"].append(quest_name)
        mem["journal"].append(f"[{timestamp}] Completed quest: {quest_name}")

    save_memory(mem)


def process_tool_result(mem: dict, tool_name: str, result_text: str) -> None:
    """Route tool results to the appropriate memory extractor."""
    try:
        result = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return

    extractors = {
        "get_my_status": extract_from_status,
        "grind_mobs": extract_from_grind,
        "fight_until_dead": extract_from_fight,
        "scan_zone": extract_from_scan,
    }

    if tool_name in extractors:
        extractors[tool_name](mem, result)
    elif tool_name in ("quests_accept", "quests_complete"):
        extract_from_quest(mem, tool_name, result_text)


def handle_remember_command(mem: dict, args: dict) -> str:
    """Handle the local 'remember' pseudo-tool."""
    section = args.get("section", "strategies")
    content = args.get("content", "")
    action = args.get("action", "add")  # add, set, remove

    if not content:
        return "Error: 'content' is required"

    if section == "facts":
        key = args.get("key", content.split(":")[0].strip() if ":" in content else content[:20])
        if action == "remove":
            mem["facts"].pop(key, None)
            save_memory(mem)
            return f"Removed fact '{key}'"
        val = args.get("value", content.split(":", 1)[1].strip() if ":" in content else content)
        mem["facts"][key] = val
        save_memory(mem)
        return f"Saved fact: {key} = {val}"

    elif section == "strategies":
        if action == "remove":
            mem["strategies"] = [s for s in mem["strategies"] if content.lower() not in s.lower()]
        else:
            mem["strategies"].append(content)
            mem["strategies"] = mem["strategies"][-MAX_STRATEGIES:]
        save_memory(mem)
        return f"Strategy {'removed' if action == 'remove' else 'saved'}: {content}"

    elif section == "inventory_notes":
        if action == "remove":
            mem["inventory_notes"] = [n for n in mem["inventory_notes"] if content.lower() not in n.lower()]
        else:
            mem["inventory_notes"].append(content)
            mem["inventory_notes"] = mem["inventory_notes"][-10:]
        save_memory(mem)
        return f"Inventory note {'removed' if action == 'remove' else 'saved'}: {content}"

    elif section == "journal":
        timestamp = datetime.now().strftime("%H:%M")
        mem["journal"].append(f"[{timestamp}] {content}")
        mem["journal"] = mem["journal"][-MAX_JOURNAL:]
        save_memory(mem)
        return f"Journal entry added: {content}"

    return f"Unknown section: {section}"
