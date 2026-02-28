"""
Self-improving policy loop for the WoG agent.

Every N cycles, evaluates gameplay performance and uses the LLM
to rewrite the strategy section of the system prompt, optimizing
for gold income, quest completions, and progression.
"""

import random
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
_WEIGHT_EXPLORATION = 5   # reward discovering new zones

MAX_POLICY_HISTORY = 20

# Decision thresholds
_ADOPT_THRESHOLD = 30.0   # Below this score, always try a new strategy
_DECLINE_WINDOW = 2       # Consecutive declining windows before revert triggers
_EMA_ALPHA = 0.3          # EMA smoothing factor
_EPSILON = 0.15           # 15% exploration rate when performing well


def _compute_score(gold: float, quests: int, xp: float, deaths: int, zones_discovered: int = 0) -> float:
    """Weighted improvement score favoring gold, quests, and exploration."""
    return (
        gold * _WEIGHT_GOLD
        + quests * _WEIGHT_QUESTS
        + xp * _WEIGHT_XP
        + deaths * _WEIGHT_DEATHS
        + zones_discovered * _WEIGHT_EXPLORATION
    )


class PolicyEvaluator:
    """Periodically evaluates agent performance and rewrites the strategy."""

    def __init__(self, eval_interval: int = 100):
        self.eval_interval = eval_interval
        self._last_snapshot: dict | None = None

        # EMA tracking
        self._ema_score: float | None = None
        self._prev_ema_score: float | None = None
        self._consecutive_declines: int = 0

        # Best-strategy tracking
        self._best_strategy: str | None = None
        self._best_score: float = float("-inf")

        # Public attrs for wandb
        self.last_action: str = ""
        self.last_ema_score: float = 0.0
        self.last_deltas: dict = {}
        self.last_improvement_score: float = 0.0

    def _take_snapshot(self, mem: dict) -> dict:
        """Capture current metrics for delta comparison."""
        stats = mem.get("stats", {})
        return {
            "gold": stats.get("total_gold_earned", 0),
            "xp": stats.get("total_xp", 0),
            "kills": stats.get("total_kills", 0),
            "deaths": stats.get("total_deaths", 0),
            "quests": len(mem.get("quests", {}).get("completed", [])),
            "zones_discovered": len(stats.get("zone_visit_counts", {})),
            "zone_transitions": stats.get("total_zone_transitions", 0),
        }

    def _compute_deltas(self, current: dict, previous: dict) -> dict:
        """Compute performance deltas between two snapshots."""
        return {
            "gold_delta": current["gold"] - previous["gold"],
            "xp_delta": current["xp"] - previous["xp"],
            "kills_delta": current["kills"] - previous["kills"],
            "deaths_delta": current["deaths"] - previous["deaths"],
            "quests_delta": current["quests"] - previous["quests"],
            "zones_discovered_delta": current.get("zones_discovered", 0) - previous.get("zones_discovered", 0),
            "zone_transitions_delta": current.get("zone_transitions", 0) - previous.get("zone_transitions", 0),
        }

    def _decide(self, score: float) -> str:
        """Core decision logic: keep, explore, revert, or adopt.

        - score >= threshold AND not declining 2+ windows -> keep (85%) or explore (15%)
        - declining 2+ windows AND best strategy exists -> revert
        - otherwise (poor performance) -> adopt new strategy
        """
        if score >= _ADOPT_THRESHOLD and self._consecutive_declines < _DECLINE_WINDOW:
            if random.random() < _EPSILON:
                return "explore"
            return "keep"

        if self._consecutive_declines >= _DECLINE_WINDOW and self._best_strategy is not None:
            return "revert"

        return "adopt"

    def _generate_new_strategy(
        self, meta_prompt: str, model, tokenizer,
    ) -> str | None:
        """Call the LLM to generate a new strategy string."""
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
            return None

        return self._parse_strategy(response)

    def _record_history(
        self,
        mem: dict,
        cycle: int,
        old_strategy: str,
        new_strategy: str,
        deltas: dict,
        score: float,
        action: str,
        ema_score: float,
    ) -> None:
        """Record policy change in mem['policy_history'] and mem['journal']."""
        if "policy_history" not in mem:
            mem["policy_history"] = []

        timestamp = datetime.now().isoformat()
        entry = {
            "cycle": cycle,
            "timestamp": timestamp,
            "old_strategy": old_strategy,
            "new_strategy": new_strategy,
            "deltas": deltas,
            "improvement_score": score,
            "action": action,
            "ema_score": ema_score,
        }
        mem["policy_history"].append(entry)
        mem["policy_history"] = mem["policy_history"][-MAX_POLICY_HISTORY:]

        if "journal" not in mem:
            mem["journal"] = []
        ts = datetime.now().strftime("%H:%M")
        mem["journal"].append(
            f"[{ts}] Policy {action} (score={score:.1f} ema={ema_score:.1f}): {new_strategy[:80]}..."
        )
        mem["journal"] = mem["journal"][-30:]

    def _build_meta_prompt(
        self,
        deltas: dict,
        current_strategy: str,
        mem: dict,
        tool_counts: dict,
        cycle: int,
    ) -> str:
        """Build a richer meta-prompt with performance, EMA trend, character state,
        tool distribution, journal entries, and recent strategy history."""
        # EMA trend
        if self._ema_score is not None:
            trend = "improving" if self._consecutive_declines == 0 else "declining"
            ema_line = f"EMA score: {self._ema_score:.1f} ({trend}, {self._consecutive_declines} consecutive decline(s))\n"
        else:
            ema_line = "EMA score: N/A (first evaluation)\n"

        # Character state from mem["facts"]
        facts = mem.get("facts", {})
        char_lines = []
        for key in ("level", "zone", "hp", "gold", "status"):
            val = facts.get(key)
            if val is not None and val != "?":
                char_lines.append(f"  {key}: {val}")
        char_block = "\n".join(char_lines) if char_lines else "  (no character data)"

        # Top 5 tool calls
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        tool_lines = ", ".join(f"{n}={c}" for n, c in top_tools) if top_tools else "(none)"

        # Last 3 journal entries
        journal = mem.get("journal", [])[-3:]
        journal_block = "\n".join(f"  {j}" for j in journal) if journal else "  (none)"

        # Last 3 strategy attempts with scores
        history = mem.get("policy_history", [])[-3:]
        hist_lines = []
        for h in history:
            s = h.get("improvement_score", 0)
            act = h.get("action", "adopted")
            strat = h.get("new_strategy", "")[:60]
            hist_lines.append(f"  [{act}] score={s:.1f}: {strat}...")
        hist_block = "\n".join(hist_lines) if hist_lines else "  (none)"

        # Zone visit distribution
        zone_visits = mem.get("stats", {}).get("zone_visit_counts", {})
        top_zones = sorted(zone_visits.items(), key=lambda x: x[1], reverse=True)[:5]
        zone_lines = ", ".join(f"{z}={c}" for z, c in top_zones) if top_zones else "(none)"

        # Recent movement path
        zone_history = mem.get("zone_history", [])[-5:]
        movement_path = " -> ".join(zh["zone"] for zh in zone_history) if zone_history else "(no movement)"

        return (
            "You are a gameplay strategy optimizer for a fantasy MMORPG agent.\n"
            f"Cycle: {cycle}\n\n"
            f"Over the last {self.eval_interval} cycles, performance was:\n"
            f"- Gold earned: {deltas['gold_delta']}\n"
            f"- XP gained: {deltas['xp_delta']}\n"
            f"- Mobs killed: {deltas['kills_delta']}\n"
            f"- Deaths: {deltas['deaths_delta']}\n"
            f"- Quests completed: {deltas['quests_delta']}\n"
            f"- New zones discovered: {deltas.get('zones_discovered_delta', 0)}\n"
            f"- Zone transitions: {deltas.get('zone_transitions_delta', 0)}\n"
            f"{ema_line}\n"
            f"Character state:\n{char_block}\n\n"
            f"Top tool usage: {tool_lines}\n"
            f"Top zones by time: {zone_lines}\n"
            f"Recent movement: {movement_path}\n\n"
            f"Recent journal:\n{journal_block}\n\n"
            f"Recent strategy history:\n{hist_block}\n\n"
            f"Current strategy:\n{current_strategy}\n\n"
            "Write an improved strategy (2-4 sentences) that maximizes gold income, "
            "quest completions, XP, and exploration while minimizing deaths. "
            "Consider whether the agent should explore new zones or farm the current one. "
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

    def maybe_update(
        self, cycle: int, mem: dict, model, tokenizer, tool_counts: dict | None = None,
    ) -> str | None:
        """Called every cycle. Returns new strategy string or None.

        Only evaluates at eval_interval boundaries. On the first call,
        just takes a baseline snapshot.
        """
        if cycle % self.eval_interval != 0:
            return None

        current = self._take_snapshot(mem)

        if self._last_snapshot is None:
            self._last_snapshot = current
            return None

        print(f"[policy] Evaluating performance at cycle {cycle}...")

        deltas = self._compute_deltas(current, self._last_snapshot)
        current_strategy = get_current_strategy(mem)

        score = _compute_score(
            deltas["gold_delta"],
            deltas["quests_delta"],
            deltas["xp_delta"],
            deltas["deaths_delta"],
            deltas.get("zones_discovered_delta", 0),
        )

        # Update EMA
        if self._ema_score is None:
            self._ema_score = score
        else:
            self._prev_ema_score = self._ema_score
            self._ema_score = _EMA_ALPHA * score + (1 - _EMA_ALPHA) * self._ema_score
            if self._ema_score < self._prev_ema_score:
                self._consecutive_declines += 1
            else:
                self._consecutive_declines = 0

        # Track best strategy
        if score > self._best_score:
            self._best_score = score
            self._best_strategy = current_strategy

        action = self._decide(score)

        if tool_counts is None:
            tool_counts = {}

        new_strategy: str | None = None

        if action == "keep":
            print(f"[policy] Keeping current strategy (score={score:.1f} ema={self._ema_score:.1f})")
            self._last_snapshot = current
            self.last_deltas = deltas
            self.last_improvement_score = score
            self.last_action = action
            self.last_ema_score = self._ema_score
            return None

        if action == "revert":
            new_strategy = self._best_strategy
            print(f"[policy] Reverting to best strategy (score={score:.1f} ema={self._ema_score:.1f} "
                  f"best={self._best_score:.1f})")
            self._consecutive_declines = 0

        elif action in ("adopt", "explore"):
            meta_prompt = self._build_meta_prompt(
                deltas, current_strategy, mem, tool_counts, cycle,
            )
            new_strategy = self._generate_new_strategy(meta_prompt, model, tokenizer)
            if new_strategy is None:
                print(f"[policy] Could not generate strategy, keeping current")
                self._last_snapshot = current
                self.last_action = "keep"
                self.last_ema_score = self._ema_score
                return None
            label = "explore" if action == "explore" else "adopt"
            print(f"[policy] {label.title()} new strategy (score={score:.1f} ema={self._ema_score:.1f}): "
                  f"{new_strategy[:100]}")

        # Record history — new_strategy is guaranteed non-None at this point
        # (revert only fires when _best_strategy is not None; adopt/explore
        # return early on generation failure)
        assert new_strategy is not None
        self._record_history(
            mem, cycle, current_strategy, new_strategy,
            deltas, score, action, self._ema_score,
        )

        self._last_snapshot = current
        self.last_deltas = deltas
        self.last_improvement_score = score
        self.last_action = action
        self.last_ema_score = self._ema_score

        return new_strategy


def get_current_strategy(mem: dict) -> str:
    """Get the latest strategy from policy_history, or fall back to default."""
    history = mem.get("policy_history", [])
    if history:
        return history[-1].get("new_strategy", DEFAULT_STRATEGY)
    return DEFAULT_STRATEGY
