"""
Trajectory logger for fine-tuning data collection.

Captures every game loop cycle as a training example:
  - Input: full ChatML prompt (system + history)
  - Output: model response, parsed tool call, MCP result
  - Reward signals: gold/XP/kills/deaths deltas
  - Writes JSONL to data/raw/traj_<timestamp>.jsonl
"""

import json
import os
import time
from datetime import datetime


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
    ) -> None:
        """Start recording a cycle. Call before inference."""
        self._current = {
            "cycle": cycle,
            "timestamp": time.time(),
            "messages": messages.copy(),
            "prompt": prompt,
            "stats_before": stats_snapshot.copy(),
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
    ) -> None:
        """Finish recording a cycle. Call after tool execution."""
        if self._current is None:
            return

        before = self._current["stats_before"]
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
            "reward_signals": {
                "gold_delta": stats_after.get("total_gold_earned", 0) - before.get("total_gold_earned", 0),
                "xp_delta": stats_after.get("total_xp", 0) - before.get("total_xp", 0),
                "kills_delta": stats_after.get("total_kills", 0) - before.get("total_kills", 0),
                "deaths_delta": stats_after.get("total_deaths", 0) - before.get("total_deaths", 0),
            },
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
