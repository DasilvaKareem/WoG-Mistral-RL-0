"""
Self-improving policy loop for the WoG agent.

Every N cycles, evaluates gameplay performance and uses the LLM
to rewrite the strategy section of the system prompt, optimizing
for gold income, quest completions, and progression.
"""

import re
from datetime import datetime

from mlx_lm import generate as mlx_generate

DEFAULT_STRATEGY = (
    "Strategy: check status first, heal if HP < 30%, fight mobs, do quests, gather, explore.\n"
    "Use grind_mobs for efficient combat. Use scan_zone to find targets."
)

# Improvement score weights
_WEIGHT_GOLD = 3
_WEIGHT_QUESTS = 50
_WEIGHT_XP = 0.1
_WEIGHT_DEATHS = -10

MAX_POLICY_HISTORY = 20


def _compute_score(gold: float, quests: int, xp: float, deaths: int) -> float:
    """Weighted improvement score favoring gold and quests."""
    return gold * _WEIGHT_GOLD + quests * _WEIGHT_QUESTS + xp * _WEIGHT_XP + deaths * _WEIGHT_DEATHS


class PolicyEvaluator:
    """Periodically evaluates agent performance and rewrites the strategy."""

    def __init__(self, eval_interval: int = 100):
        self.eval_interval = eval_interval
        self._last_snapshot: dict | None = None

    def _take_snapshot(self, mem: dict) -> dict:
        """Capture current metrics for delta comparison."""
        stats = mem.get("stats", {})
        return {
            "gold": stats.get("total_gold_earned", 0),
            "xp": stats.get("total_xp", 0),
            "kills": stats.get("total_kills", 0),
            "deaths": stats.get("total_deaths", 0),
            "quests": len(mem.get("quests", {}).get("completed", [])),
        }

    def _compute_deltas(self, current: dict, previous: dict) -> dict:
        """Compute performance deltas between two snapshots."""
        return {
            "gold_delta": current["gold"] - previous["gold"],
            "xp_delta": current["xp"] - previous["xp"],
            "kills_delta": current["kills"] - previous["kills"],
            "deaths_delta": current["deaths"] - previous["deaths"],
            "quests_delta": current["quests"] - previous["quests"],
        }

    def _build_meta_prompt(self, deltas: dict, current_strategy: str) -> str:
        """Build a concise meta-prompt for the LLM to evaluate and rewrite strategy."""
        return (
            "You are a gameplay strategy optimizer for a fantasy MMORPG agent.\n"
            f"Over the last {self.eval_interval} cycles, performance was:\n"
            f"- Gold earned: {deltas['gold_delta']}\n"
            f"- XP gained: {deltas['xp_delta']}\n"
            f"- Mobs killed: {deltas['kills_delta']}\n"
            f"- Deaths: {deltas['deaths_delta']}\n"
            f"- Quests completed: {deltas['quests_delta']}\n\n"
            f"Current strategy:\n{current_strategy}\n\n"
            "Write an improved strategy (2-4 sentences) that maximizes gold income, "
            "quest completions, and XP while minimizing deaths. "
            "Wrap your answer in <strategy></strategy> tags."
        )

    def _parse_strategy(self, response: str) -> str | None:
        """Extract strategy from <strategy> tags, or use raw text if short."""
        m = re.search(r"<strategy>(.*?)</strategy>", response, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if 10 < len(text) < 500:
                return text
        # Fallback: use raw text if reasonably sized
        cleaned = response.strip()
        if 10 < len(cleaned) < 300:
            return cleaned
        return None

    def maybe_update(self, cycle: int, mem: dict, model, tokenizer) -> str | None:
        """Called every cycle. Returns new strategy string or None.

        Only evaluates at eval_interval boundaries. On the first call,
        just takes a baseline snapshot.
        """
        if cycle % self.eval_interval != 0:
            return None

        current = self._take_snapshot(mem)

        if self._last_snapshot is None:
            # First evaluation — just record baseline
            self._last_snapshot = current
            return None

        print(f"[policy] Evaluating performance at cycle {cycle}...")

        deltas = self._compute_deltas(current, self._last_snapshot)
        current_strategy = get_current_strategy(mem)

        improvement_score = _compute_score(
            deltas["gold_delta"],
            deltas["quests_delta"],
            deltas["xp_delta"],
            deltas["deaths_delta"],
        )

        meta_prompt = self._build_meta_prompt(deltas, current_strategy)

        try:
            response = mlx_generate(
                model,
                tokenizer,
                prompt=meta_prompt,
                max_tokens=200,
                verbose=False,
            )
        except Exception as e:
            print(f"[policy] LLM generation failed: {e}")
            self._last_snapshot = current
            return None

        new_strategy = self._parse_strategy(response)

        if new_strategy is None:
            print(f"[policy] Could not parse strategy from response: {response[:100]}")
            self._last_snapshot = current
            return None

        # Record in policy history
        if "policy_history" not in mem:
            mem["policy_history"] = []

        timestamp = datetime.now().isoformat()
        entry = {
            "cycle": cycle,
            "timestamp": timestamp,
            "old_strategy": current_strategy,
            "new_strategy": new_strategy,
            "deltas": deltas,
            "improvement_score": improvement_score,
        }
        mem["policy_history"].append(entry)
        mem["policy_history"] = mem["policy_history"][-MAX_POLICY_HISTORY:]

        # Add journal entry
        if "journal" not in mem:
            mem["journal"] = []
        ts = datetime.now().strftime("%H:%M")
        mem["journal"].append(
            f"[{ts}] Policy updated (score={improvement_score:.1f}): {new_strategy[:80]}..."
        )
        mem["journal"] = mem["journal"][-30:]

        self._last_snapshot = current
        self.last_deltas = deltas
        self.last_improvement_score = improvement_score

        print(f"[policy] New strategy (score={improvement_score:.1f}): {new_strategy[:100]}")
        return new_strategy


def get_current_strategy(mem: dict) -> str:
    """Get the latest strategy from policy_history, or fall back to default."""
    history = mem.get("policy_history", [])
    if history:
        return history[-1].get("new_strategy", DEFAULT_STRATEGY)
    return DEFAULT_STRATEGY
