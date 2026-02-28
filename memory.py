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
        "quests": {"completed": [], "active": [], "available": []},
        "zones": {},
        "zone_history": [],          # chronological list of zone visits
        "strategies": [],
        "inventory_notes": [],
        "journal": [],
        "stats": {
            "total_kills": 0,
            "total_deaths": 0,
            "total_xp": 0,
            "total_gold_earned": 0,
            "total_quests_xp": 0,
            "total_quests_gold": 0,
            "total_zone_transitions": 0,
            "zone_visit_counts": {},   # {zone_id: visit_count}
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
    if quests.get("available"):
        lines.append("\n### Available Quests")
        for q in quests["available"][-5:]:
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

    # Movement history (last 5 transitions)
    zone_history = mem.get("zone_history", [])
    if zone_history:
        lines.append(f"\n### Recent Movement ({mem.get('stats', {}).get('total_zone_transitions', 0)} total transitions)")
        for zh in zone_history[-5:]:
            if zh.get("from"):
                lines.append(f"- {zh['from']} -> {zh['zone']} ({zh.get('time', '?')})")
            else:
                lines.append(f"- Arrived at {zh['zone']} ({zh.get('time', '?')})")

    # Lifetime stats
    stats = mem.get("stats", {})
    if stats:
        lines.append(f"\n### Lifetime Stats")
        lines.append(f"- Kills: {stats.get('total_kills', 0)} | "
                     f"Deaths: {stats.get('total_deaths', 0)} | "
                     f"XP: {stats.get('total_xp', 0)} | "
                     f"Gold earned: {stats.get('total_gold_earned', 0)} | "
                     f"Sessions: {stats.get('sessions', 0)}")
        lines.append(f"- Zones visited: {len(stats.get('zone_visit_counts', {}))} | "
                     f"Zone transitions: {stats.get('total_zone_transitions', 0)} | "
                     f"Quest XP: {stats.get('total_quests_xp', 0)} | "
                     f"Quest gold: {stats.get('total_quests_gold', 0)}")

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


MAX_ZONE_HISTORY = 50


def extract_from_travel(mem: dict, tool_name: str, result: dict) -> None:
    """Update memory from travel/transition/move results."""
    if not isinstance(result, dict):
        return
    timestamp = datetime.now().strftime("%H:%M")

    # Try common field names the MCP server might return
    new_zone = (
        result.get("zone")
        or result.get("zoneId")
        or result.get("destination")
        or result.get("newZone")
        or result.get("arrivedAt")
    )
    old_zone = (
        result.get("fromZone")
        or result.get("previousZone")
        or result.get("origin")
        or mem.get("facts", {}).get("zone")
    )
    travel_time = result.get("travelTime") or result.get("duration")
    method = result.get("method") or result.get("type") or tool_name

    if new_zone:
        mem["facts"]["zone"] = new_zone

        # Track zone visit counts
        counts = mem["stats"].setdefault("zone_visit_counts", {})
        counts[new_zone] = counts.get(new_zone, 0) + 1

        # Track total transitions
        mem["stats"]["total_zone_transitions"] = mem["stats"].get("total_zone_transitions", 0) + 1

        # Chronological zone history
        history = mem.setdefault("zone_history", [])
        history.append({
            "zone": new_zone,
            "from": old_zone,
            "time": timestamp,
            "method": method,
        })
        mem["zone_history"] = history[-MAX_ZONE_HISTORY:]

        # Update zone exploration map
        if new_zone not in mem.get("zones", {}):
            mem.setdefault("zones", {})[new_zone] = "visited (not yet scanned)"

        # Journal
        if old_zone and old_zone != new_zone:
            entry = f"[{timestamp}] Traveled {old_zone} -> {new_zone}"
        else:
            entry = f"[{timestamp}] Arrived at {new_zone}"
        if travel_time:
            entry += f" ({travel_time}s)"
        mem["journal"].append(entry)
        mem["journal"] = mem["journal"][-MAX_JOURNAL:]

    # Handle failed travel
    elif result.get("error") or result.get("failed"):
        reason = result.get("error") or result.get("reason") or "unknown"
        mem["journal"].append(f"[{timestamp}] Travel failed: {reason}")
        mem["journal"] = mem["journal"][-MAX_JOURNAL:]

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
        # Store richer quest data if available
        quest_entry = quest_name
        objectives = result.get("objectives") or result.get("description")
        rewards = result.get("rewards") or result.get("reward")
        if objectives or rewards:
            parts = [quest_name]
            if objectives:
                parts.append(f"obj: {objectives}" if isinstance(objectives, str) else f"obj: {json.dumps(objectives)}")
            if rewards:
                parts.append(f"rewards: {rewards}" if isinstance(rewards, str) else f"rewards: {json.dumps(rewards)}")
            quest_entry = " | ".join(parts)
        if quest_entry not in mem["quests"]["active"]:
            mem["quests"]["active"].append(quest_entry)
        mem["journal"].append(f"[{timestamp}] Accepted quest: {quest_entry}")

    elif tool_name == "quests_complete" and isinstance(result, dict):
        quest_name = result.get("questId") or result.get("name") or "unknown quest"
        # Remove from active (match on quest ID prefix)
        mem["quests"]["active"] = [
            q for q in mem["quests"]["active"]
            if not q.startswith(quest_name)
        ]
        if quest_name not in mem["quests"]["completed"]:
            mem["quests"]["completed"].append(quest_name)

        # Track quest rewards in stats
        xp_reward = result.get("xpReward") or result.get("xpGained") or 0
        gold_reward = result.get("goldReward") or result.get("goldGained") or result.get("goldEarned") or 0
        if isinstance(xp_reward, (int, float)) and xp_reward > 0:
            mem["stats"]["total_xp"] = mem["stats"].get("total_xp", 0) + xp_reward
            mem["stats"]["total_quests_xp"] = mem["stats"].get("total_quests_xp", 0) + xp_reward
        if isinstance(gold_reward, (int, float)) and gold_reward > 0:
            mem["stats"]["total_gold_earned"] = mem["stats"].get("total_gold_earned", 0) + gold_reward
            mem["stats"]["total_quests_gold"] = mem["stats"].get("total_quests_gold", 0) + gold_reward

        reward_str = ""
        if xp_reward:
            reward_str += f" +{xp_reward} XP"
        if gold_reward:
            reward_str += f" +{gold_reward} gold"
        mem["journal"].append(f"[{timestamp}] Completed quest: {quest_name}{reward_str}")

    elif tool_name == "quests_list" and isinstance(result, (dict, list)):
        # Track available quests so the agent knows what's out there
        quests_list = result if isinstance(result, list) else result.get("quests", result.get("available", []))
        if isinstance(quests_list, list):
            available = []
            for q in quests_list[:10]:
                if isinstance(q, dict):
                    qid = q.get("questId") or q.get("id") or q.get("name") or "?"
                    desc = q.get("description", "")[:60]
                    level = q.get("level") or q.get("minLevel")
                    entry = qid
                    if desc:
                        entry += f": {desc}"
                    if level:
                        entry += f" (L{level}+)"
                    available.append(entry)
                elif isinstance(q, str):
                    available.append(q)
            mem["quests"]["available"] = available
            mem["journal"].append(f"[{timestamp}] Found {len(available)} available quests")

    elif tool_name == "quests_progress" and isinstance(result, dict):
        quest_name = result.get("questId") or result.get("name") or "unknown quest"
        progress = result.get("progress") or result.get("percentage") or result.get("completion")
        objectives = result.get("objectives") or result.get("status")
        entry = f"[{timestamp}] Quest progress: {quest_name}"
        if progress is not None:
            entry += f" {progress}%"
        if objectives:
            entry += f" — {objectives}" if isinstance(objectives, str) else f" — {json.dumps(objectives)}"
        mem["journal"].append(entry)

    elif tool_name == "quests_abandon" and isinstance(result, dict):
        quest_name = result.get("questId") or result.get("name") or "unknown quest"
        mem["quests"]["active"] = [
            q for q in mem["quests"]["active"]
            if not q.startswith(quest_name)
        ]
        mem["journal"].append(f"[{timestamp}] Abandoned quest: {quest_name}")

    mem["journal"] = mem["journal"][-MAX_JOURNAL:]
    save_memory(mem)


_TRAVEL_PREFIXES = ("travel", "transition", "move_")
_QUEST_TOOLS = ("quests_accept", "quests_complete", "quests_list", "quests_progress", "quests_abandon")


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
    elif tool_name in _QUEST_TOOLS:
        extract_from_quest(mem, tool_name, result_text)
    elif any(tool_name.startswith(p) for p in _TRAVEL_PREFIXES):
        extract_from_travel(mem, tool_name, result)


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
