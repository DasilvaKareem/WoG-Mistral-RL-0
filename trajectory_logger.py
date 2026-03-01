"""
Trajectory logger for fine-tuning data collection.

Captures every game loop cycle as a training example:
  - Input: full ChatML prompt (system + history)
  - Output: model response, parsed tool call, MCP result
  - Reward signals: all stat deltas + composite reward score
  - Writes JSONL locally AND uploads to Firebase Storage on every flush
    so data survives container crashes.

Firebase env vars (from Modal 'firebase-admin' secret):
  FIREBASE_SERVICE_ACCOUNT_JSON  — service account JSON as string
  FIREBASE_STORAGE_BUCKET        — e.g. my-project.appspot.com
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

# Upload to Firebase every N records (not every single flush — batches are cheaper)
FIREBASE_UPLOAD_INTERVAL = 5


def _init_firebase_bucket():
    """Try to initialize Firebase Storage bucket. Returns bucket or None."""
    try:
        import firebase_admin
        from firebase_admin import credentials, storage

        # Try multiple possible key names across different secrets
        sa_json = (
            os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            or os.environ.get("SERVICE_ACCOUNT_JSON")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            or os.environ.get("FIREBASE_CREDENTIALS")
        )
        bucket_name = (
            os.environ.get("FIREBASE_STORAGE_BUCKET")
            or os.environ.get("STORAGE_BUCKET")
            or os.environ.get("FIREBASE_BUCKET")
            or os.environ.get("GCS_BUCKET")
        )

        # Debug: print which firebase-related env vars are present
        firebase_keys = [k for k in os.environ if "FIREBASE" in k.upper() or "GCS" in k.upper() or "GOOGLE" in k.upper()]
        print(f"[trajectory] Firebase-related env vars found: {firebase_keys}")

        if not sa_json or not bucket_name:
            print(f"[trajectory] Firebase env vars missing (sa_json={'set' if sa_json else 'MISSING'}, bucket={'set' if bucket_name else 'MISSING'}) — local-only logging")
            return None

        import base64
        try:
            sa_dict = json.loads(sa_json)
        except json.JSONDecodeError:
            sa_dict = json.loads(base64.b64decode(sa_json).decode("utf-8"))

        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_dict)
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        bucket = storage.bucket()
        print(f"[trajectory] Firebase Storage ready: gs://{bucket_name}")
        return bucket
    except Exception as e:
        print(f"[trajectory] Firebase init failed, local-only: {e}")
        return None


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

    def __init__(self, output_dir: str = "data/raw", agent_id: str = "0"):
        self.output_dir = output_dir
        self.agent_id = agent_id
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"traj_agent{agent_id}_{timestamp}.jsonl"
        self.filepath = os.path.join(output_dir, self.filename)
        self._file = open(self.filepath, "a")
        self._current: dict | None = None
        self._total_records = 0
        self._last_upload_count = 0

        self._bucket = _init_firebase_bucket()
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

        new_times = times_after[len(times_before):]
        last_quest_time_s = new_times[-1] if new_times else None

        reward_signals = {
            "gold_delta": stats_after.get("total_gold_earned", 0) - before.get("total_gold_earned", 0),
            "xp_delta": stats_after.get("total_xp", 0) - before.get("total_xp", 0),
            "kills_delta": stats_after.get("total_kills", 0) - before.get("total_kills", 0),
            "deaths_delta": stats_after.get("total_deaths", 0) - before.get("total_deaths", 0),
            "quests_completed_delta": quests_completed - self._current["quests_completed_before"],
            "quest_gold_delta": stats_after.get("total_quests_gold", 0) - before.get("total_quests_gold", 0),
            "quest_xp_delta": stats_after.get("total_quests_xp", 0) - before.get("total_quests_xp", 0),
            "quest_completion_time_s": last_quest_time_s,
            "quest_difficulty": last_quest_difficulty,
            "zones_discovered_delta": zones_discovered - self._current["zones_discovered_before"],
            "zone_transitions_delta": (
                stats_after.get("total_zone_transitions", 0) - before.get("total_zone_transitions", 0)
            ),
            "zone_before": self._current["zone_before"],
            "zone_after": zone,
        }

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

        # Upload to Firebase every FIREBASE_UPLOAD_INTERVAL records
        if self._bucket and (self._total_records - self._last_upload_count) >= FIREBASE_UPLOAD_INTERVAL:
            self._upload_to_firebase()

    def _upload_to_firebase(self) -> None:
        """Upload the current JSONL file to Firebase Storage."""
        if not self._bucket:
            return
        try:
            self._file.flush()
            blob_path = f"trajectories/agent{self.agent_id}/{self.filename}"
            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(self.filepath)
            self._last_upload_count = self._total_records
            print(f"[trajectory] Uploaded {self._total_records} records to gs://{blob_path}")
        except Exception as e:
            print(f"[trajectory] Firebase upload failed (data safe locally): {e}")

    def get_stats(self) -> dict:
        """Return logging statistics."""
        return {
            "filepath": self.filepath,
            "total_records": self._total_records,
        }

    def close(self) -> None:
        """Close the JSONL file and do a final upload."""
        if self._file and not self._file.closed:
            self._file.close()
            print(f"[trajectory] Saved {self._total_records} records to {self.filepath}")
        if self._bucket and self._total_records > self._last_upload_count:
            self._upload_to_firebase()
