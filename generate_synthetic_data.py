"""
Generate synthetic trajectory data for pipeline testing.

Produces realistic JSONL records mimicking what the live agent would
capture, so the full train → eval pipeline can run without a game server.
"""

import json
import os
import random
import time

TOOLS_JSON = json.dumps([
    {"type":"function","function":{"name":"get_my_status","description":"Check your character status","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"scan_zone","description":"Scan current zone for mobs, resources, players","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"grind_mobs","description":"Fight mobs in current zone automatically","parameters":{"type":"object","properties":{"rounds":{"type":"integer","description":"Number of rounds"}},"required":["rounds"]}}},
    {"type":"function","function":{"name":"fight_until_dead","description":"Fight a specific mob until it dies","parameters":{"type":"object","properties":{"targetId":{"type":"string"}},"required":["targetId"]}}},
    {"type":"function","function":{"name":"gather_ore","description":"Gather ore from nearby nodes","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"gather_flowers","description":"Gather flowers from nearby nodes","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"heal","description":"Use a healing item or rest to recover HP","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"rest","description":"Rest to recover HP slowly","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"shop_buy","description":"Buy an item from the shop","parameters":{"type":"object","properties":{"itemId":{"type":"string"},"quantity":{"type":"integer"}},"required":["itemId"]}}},
    {"type":"function","function":{"name":"shop_sell","description":"Sell an item to the shop","parameters":{"type":"object","properties":{"itemId":{"type":"string"},"quantity":{"type":"integer"}},"required":["itemId"]}}},
    {"type":"function","function":{"name":"equip_item","description":"Equip an item from inventory","parameters":{"type":"object","properties":{"itemId":{"type":"string"}},"required":["itemId"]}}},
    {"type":"function","function":{"name":"travel","description":"Travel to another zone","parameters":{"type":"object","properties":{"destination":{"type":"string"}},"required":["destination"]}}},
    {"type":"function","function":{"name":"quests_list","description":"List available and active quests","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"quests_accept","description":"Accept a quest","parameters":{"type":"object","properties":{"questId":{"type":"string"}},"required":["questId"]}}},
    {"type":"function","function":{"name":"inventory","description":"Check your inventory","parameters":{"type":"object","properties":{}}}},
])

SYSTEM_PROMPT = (
    "You are a function calling AI model. You are an autonomous agent playing WoG, a fantasy MMORPG.\n"
    "You are provided with function signatures within <tools></tools> XML tags.\n"
    "You may call one or more functions to assist with gameplay. Always call a function.\n"
    "Don't make assumptions about what values to plug into functions.\n\n"
    "Here are the available tools:\n<tools>\n" + TOOLS_JSON + "\n</tools>\n\n"
    "For each function call return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": "function_name", "arguments": {"key": "value"}}\n</tool_call>\n\n'
    "Strategy: check status first, heal if HP < 30%, fight mobs, do quests, gather, explore.\n"
    "Use grind_mobs for efficient combat. Use scan_zone to find targets."
)

TOOL_ACTIONS = [
    {"name": "get_my_status", "arguments": {}, "weight": 15},
    {"name": "scan_zone", "arguments": {}, "weight": 10},
    {"name": "grind_mobs", "arguments": {"rounds": 3}, "weight": 25},
    {"name": "grind_mobs", "arguments": {"rounds": 5}, "weight": 15},
    {"name": "fight_until_dead", "arguments": {"targetId": "mob_wolf_01"}, "weight": 10},
    {"name": "fight_until_dead", "arguments": {"targetId": "mob_goblin_03"}, "weight": 8},
    {"name": "gather_ore", "arguments": {}, "weight": 5},
    {"name": "gather_flowers", "arguments": {}, "weight": 3},
    {"name": "heal", "arguments": {}, "weight": 8},
    {"name": "rest", "arguments": {}, "weight": 3},
    {"name": "shop_buy", "arguments": {"itemId": "health_potion", "quantity": 3}, "weight": 3},
    {"name": "shop_sell", "arguments": {"itemId": "wolf_pelt", "quantity": 5}, "weight": 3},
    {"name": "travel", "arguments": {"destination": "dark-forest"}, "weight": 3},
    {"name": "quests_list", "arguments": {}, "weight": 4},
    {"name": "quests_accept", "arguments": {"questId": "kill_wolves_01"}, "weight": 3},
    {"name": "inventory", "arguments": {}, "weight": 4},
    {"name": "equip_item", "arguments": {"itemId": "iron_sword"}, "weight": 2},
]

USER_PROMPTS = [
    "Decide your next action. Always call a tool.",
    "<tool_response>\n{\"level\":5,\"hp\":45,\"maxHp\":100,\"zone\":\"village-square\",\"goldBalance\":230}\n</tool_response>\n\nDecide your next action. Always call a tool.",
    "<tool_response>\n{\"summary\":{\"mobsKilled\":3,\"totalXpGained\":45,\"goldEarned\":12}}\n</tool_response>\n\nDecide your next action. Always call a tool.",
    "<tool_response>\n{\"mobs\":{\"count\":5},\"oreNodes\":{\"count\":2}}\n</tool_response>\n\nDecide your next action. Always call a tool.",
    "<tool_response>\n{\"healed\":true,\"hp\":100,\"maxHp\":100}\n</tool_response>\n\nDecide your next action. Always call a tool.",
    "<tool_response>\n{\"killed\":true,\"target\":{\"name\":\"Wolf\",\"level\":3},\"xpGained\":20,\"goldEarned\":5}\n</tool_response>\n\nDecide your next action. Always call a tool.",
]


def pick_action() -> dict:
    weights = [a["weight"] for a in TOOL_ACTIONS]
    return random.choices(TOOL_ACTIONS, weights=weights, k=1)[0]


def make_response(action: dict) -> str:
    call_json = json.dumps({"name": action["name"], "arguments": action["arguments"]})
    return f"I'll {action['name'].replace('_', ' ')} now.\n<tool_call>\n{call_json}\n</tool_call>"


def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "traj_synthetic.jsonl")

    num_records = 800
    stats = {
        "total_kills": 0,
        "total_deaths": 0,
        "total_xp": 0,
        "total_gold_earned": 0,
        "sessions": 1,
    }

    records = []
    for cycle in range(1, num_records + 1):
        action = pick_action()
        response = make_response(action)
        user_msg = random.choice(USER_PROMPTS)

        stats_before = dict(stats)

        # Simulate reward signals
        gold_delta = 0
        xp_delta = 0
        kills_delta = 0
        deaths_delta = 0
        tool_success = True

        if action["name"] == "grind_mobs":
            rounds = action["arguments"].get("rounds", 3)
            kills_delta = random.randint(1, rounds)
            xp_delta = kills_delta * random.randint(10, 25)
            gold_delta = kills_delta * random.randint(2, 8)
            deaths_delta = 1 if random.random() < 0.05 else 0
        elif action["name"] == "fight_until_dead":
            kills_delta = 1 if random.random() < 0.85 else 0
            xp_delta = random.randint(15, 35) if kills_delta else 0
            gold_delta = random.randint(3, 10) if kills_delta else 0
            deaths_delta = 1 if random.random() < 0.08 else 0
        elif action["name"] in ("gather_ore", "gather_flowers"):
            gold_delta = random.randint(1, 5)
            xp_delta = random.randint(2, 8)
        elif action["name"] == "shop_sell":
            gold_delta = random.randint(5, 20)
        elif action["name"] == "shop_buy":
            gold_delta = -random.randint(5, 15)

        # 5% chance of tool error
        if random.random() < 0.05:
            tool_success = False

        stats["total_kills"] += kills_delta
        stats["total_xp"] += xp_delta
        stats["total_gold_earned"] += max(gold_delta, 0)
        stats["total_deaths"] += deaths_delta

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        record = {
            "cycle": cycle,
            "timestamp": time.time() + cycle * 3,
            "messages": messages,
            "prompt": prompt,
            "response": response,
            "tool_call": {"name": action["name"], "arguments": action["arguments"]},
            "tool_name": action["name"],
            "tool_args": action["arguments"],
            "tool_result": json.dumps({"success": True}),
            "tool_success": tool_success,
            "inference_time": random.uniform(0.8, 2.5),
            "reward_signals": {
                "gold_delta": gold_delta,
                "xp_delta": xp_delta,
                "kills_delta": kills_delta,
                "deaths_delta": deaths_delta,
            },
            "stats_before": stats_before,
            "stats_after": dict(stats),
        }
        records.append(record)

    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} synthetic trajectory records -> {filepath}")
    print(f"Final stats: {json.dumps(stats, indent=2)}")

    # Stats breakdown
    success = sum(1 for r in records if r["tool_success"])
    has_tool = sum(1 for r in records if r["tool_name"])
    no_death = sum(1 for r in records if r["reward_signals"]["deaths_delta"] == 0)
    filterable = sum(1 for r in records
                     if r["tool_success"] and r["tool_name"]
                     and r["reward_signals"]["deaths_delta"] == 0)
    print(f"Success: {success}/{len(records)}, Has tool: {has_tool}/{len(records)}, "
          f"No death: {no_death}/{len(records)}, Filterable: {filterable}/{len(records)}")


if __name__ == "__main__":
    main()
