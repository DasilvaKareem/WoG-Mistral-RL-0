"""
Trajectory logger for fine-tuning data collection.

Captures every game loop cycle as a training example:
  - Input: full ChatML prompt (system + history)
  - Output: model response, parsed tool call, MCP result
  - Reward signals: all stat deltas + composite reward score
  - Writes JSONL to data/raw/traj_<timestamp>.jsonl
"""

import json
import os
import time
from datetime import datetime

# Reward weights — mirrored from policy.py so trajectories use the same scoring
REWARD_WEIGHT_GOLD = 3.0
REWARD_WEIGHT_QUESTS = 50.0
REWARD_WEIGHT_XP = 0.1
REWARD_WEIGHT_DEATHS = -10.0
REWARD_WEIGHT_EXPLORATION = 5.0
REWARD_WEIGHT_QUEST_GOLD = 3.0
REWARD_WEIGHT_QUEST_XP = 0.1


def compute_reward(signals: dict) -> float:
    """Compute a scalar reward from all reward signals.
    Same weights as policy._compute_score so RL and SFT agree on what's good."""
    return (
        signals.get("gold_delta", 0) * REWARD_WEIGHT_GOLD
        + signals.get("quests_completed_delta", 0) * REWARD_WEIGHT_QUESTS
        + signals.get("xp_delta", 0) * REWARD_WEIGHT_XP
        + signals.get("deaths_delta", 0) * REWARD_WEIGHT_DEATHS
        + signals.get("zones_discovered_delta", 0) * REWARD_WEIGHT_EXPLORATION
        + signals.get("quest_gold_delta", 0) * REWARD_WEIGHT_QUEST_GOLD
        + signals.get("quest_xp_delta", 0) * REWARD_WEIGHT_QUEST_XP
    )


class TrajectoryLogger:
    """Logs agent trajectories as JSONL for LoRA fine-tuning."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"traj_{timestamp}.jsonl")
        self._file = open(self.filepath, "a")
        self._current: dict | None = None
        self._total_records = 0
        print(f"[trajectory] Logging to {self.filepath}")

    def begin_cycle(
        self,
        cycle: int,
        messages: list[dict],
        prompt: str,
        stats_snapshot: dict,
        quests_completed: int = 0,
        zones_discovered: int = 0,
        zone: str | None = None,
        quest_completion_times: list[float] | None = None,
    ) -> None:
        """Start recording a cycle. Call before inference."""
        self._current = {
            "cycle": cycle,
            "timestamp": time.time(),
            "messages": messages.copy(),
            "prompt": prompt,
            "stats_before": stats_snapshot.copy(),
            "quests_completed_before": quests_completed,
            "zones_discovered_before": zones_discovered,
            "zone_before": zone,
            "quest_completion_times_before": list(quest_completion_times or []),
        }

    def end_cycle(
        self,
        response: str,
        tool_call: dict | None,
        tool_name: str | None,
        tool_args: dict | None,
        tool_result: str | None,
        tool_success: bool,
        stats_after: dict,
        inference_time: float,
        quests_completed: int = 0,
        zones_discovered: int = 0,
        zone: str | None = None,
        quest_completion_times: list[float] | None = None,
        last_quest_difficulty: str | None = None,
    ) -> None:
        """Finish recording a cycle. Call after tool execution."""
        if self._current is None:
            return

        before = self._current["stats_before"]
        times_after = list(quest_completion_times or [])
        times_before = self._current["quest_completion_times_before"]

        # If a quest was completed this cycle, capture its completion time
        new_times = times_after[len(times_before):]
        last_quest_time_s = new_times[-1] if new_times else None

        # Full reward signals — every metric we track
        reward_signals = {
            # Combat
            "gold_delta": stats_after.get("total_gold_earned", 0) - before.get("total_gold_earned", 0),
            "xp_delta": stats_after.get("total_xp", 0) - before.get("total_xp", 0),
            "kills_delta": stats_after.get("total_kills", 0) - before.get("total_kills", 0),
            "deaths_delta": stats_after.get("total_deaths", 0) - before.get("total_deaths", 0),
            # Quests
            "quests_completed_delta": quests_completed - self._current["quests_completed_before"],
            "quest_gold_delta": stats_after.get("total_quests_gold", 0) - before.get("total_quests_gold", 0),
            "quest_xp_delta": stats_after.get("total_quests_xp", 0) - before.get("total_quests_xp", 0),
            "quest_completion_time_s": last_quest_time_s,
            "quest_difficulty": last_quest_difficulty,
            # Exploration
            "zones_discovered_delta": zones_discovered - self._current["zones_discovered_before"],
            "zone_transitions_delta": (
                stats_after.get("total_zone_transitions", 0) - before.get("total_zone_transitions", 0)
            ),
            # Context
            "zone_before": self._current["zone_before"],
            "zone_after": zone,
        }

        # Composite scalar reward
        reward = compute_reward(reward_signals)

        record = {
            "cycle": self._current["cycle"],
            "timestamp": self._current["timestamp"],
            "messages": self._current["messages"],
            "prompt": self._current["prompt"],
            "response": response,
            "tool_call": tool_call,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "tool_success": tool_success,
            "inference_time": inference_time,
            "reward_signals": reward_signals,
            "reward": reward,
            "stats_before": before,
            "stats_after": stats_after.copy(),
        }

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._total_records += 1
        self._current = None

    def get_stats(self) -> dict:
        """Return logging statistics."""
        return {
            "filepath": self.filepath,
            "total_records": self._total_records,
        }

    def close(self) -> None:
        """Close the JSONL file."""
        if self._file and not self._file.closed:
            self._file.close()
            print(f"[trajectory] Saved {self._total_records} records to {self.filepath}")
