"""
Self-improving policy loop for the WoG agent.
NVIDIA/CUDA version — uses transformers instead of mlx_lm.

Every N cycles, evaluates gameplay performance and uses the LLM
to rewrite the strategy section of the system prompt.
"""

import random
import re
from datetime import datetime

import torch

DEFAULT_STRATEGY = (
    "Strategy: quest-chain every cycle. Check status, accept the highest-reward quest available, "
    "complete its objectives (fight/navigate/gather), then immediately complete it for rewards. "
    "After every 3 quest completions, learn new techniques and buy better equipment from shops."
)

_WEIGHT_GOLD = 3
_WEIGHT_QUESTS = 50
_WEIGHT_XP = 0.1
_WEIGHT_DEATHS = -10

MAX_POLICY_HISTORY = 20

_ADOPT_THRESHOLD = 30.0
_DECLINE_WINDOW = 2
_EMA_ALPHA = 0.3
_EPSILON = 0.15


def _compute_score(gold: float, quests: int, xp: float, deaths: int) -> float:
    return gold * _WEIGHT_GOLD + quests * _WEIGHT_QUESTS + xp * _WEIGHT_XP + deaths * _WEIGHT_DEATHS


class PolicyEvaluator:
    """Periodically evaluates agent performance and rewrites the strategy."""

    def __init__(self, eval_interval: int = 100):
        self.eval_interval = eval_interval
        self._last_snapshot: dict | None = None
        self._ema_score: float | None = None
        self._prev_ema_score: float | None = None
        self._consecutive_declines: int = 0
        self._best_strategy: str | None = None
        self._best_score: float = float("-inf")
        self.last_action: str = ""
        self.last_ema_score: float = 0.0
        self.last_deltas: dict = {}
        self.last_improvement_score: float = 0.0

    def _take_snapshot(self, mem: dict) -> dict:
        stats = mem.get("stats", {})
        return {
            "gold": stats.get("total_gold_earned", 0),
            "xp": stats.get("total_xp", 0),
            "kills": stats.get("total_kills", 0),
            "deaths": stats.get("total_deaths", 0),
            "quests": len(mem.get("quests", {}).get("completed", [])),
        }

    def _compute_deltas(self, current: dict, previous: dict) -> dict:
        return {
            "gold_delta": current["gold"] - previous["gold"],
            "xp_delta": current["xp"] - previous["xp"],
            "kills_delta": current["kills"] - previous["kills"],
            "deaths_delta": current["deaths"] - previous["deaths"],
            "quests_delta": current["quests"] - previous["quests"],
        }

    def _decide(self, score: float) -> str:
        if score >= _ADOPT_THRESHOLD and self._consecutive_declines < _DECLINE_WINDOW:
            if random.random() < _EPSILON:
                return "explore"
            return "keep"
        if self._consecutive_declines >= _DECLINE_WINDOW and self._best_strategy is not None:
            return "revert"
        return "adopt"

    def _generate_new_strategy(self, meta_prompt: str, model, tokenizer) -> str | None:
        """Call the LLM to generate a new strategy string (NVIDIA/transformers)."""
        try:
            inputs = tokenizer(meta_prompt, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        except Exception as e:
            print(f"[policy] LLM generation failed: {e}")
            return None
        return self._parse_strategy(response)

    def _record_history(self, mem, cycle, old_strategy, new_strategy, deltas, score, action, ema_score):
        if "policy_history" not in mem:
            mem["policy_history"] = []
        timestamp = datetime.now().isoformat()
        entry = {
            "cycle": cycle, "timestamp": timestamp,
            "old_strategy": old_strategy, "new_strategy": new_strategy,
            "deltas": deltas, "improvement_score": score,
            "action": action, "ema_score": ema_score,
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

    def _build_meta_prompt(self, deltas, current_strategy, mem, tool_counts, cycle):
        if self._ema_score is not None:
            trend = "improving" if self._consecutive_declines == 0 else "declining"
            ema_line = f"EMA score: {self._ema_score:.1f} ({trend}, {self._consecutive_declines} consecutive decline(s))\n"
        else:
            ema_line = "EMA score: N/A (first evaluation)\n"
        facts = mem.get("facts", {})
        char_lines = []
        for key in ("level", "zone", "hp", "gold", "status"):
            val = facts.get(key)
            if val is not None and val != "?":
                char_lines.append(f"  {key}: {val}")
        char_block = "\n".join(char_lines) if char_lines else "  (no character data)"
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        tool_lines = ", ".join(f"{n}={c}" for n, c in top_tools) if top_tools else "(none)"
        journal = mem.get("journal", [])[-3:]
        journal_block = "\n".join(f"  {j}" for j in journal) if journal else "  (none)"
        history = mem.get("policy_history", [])[-3:]
        hist_lines = []
        for h in history:
            s = h.get("improvement_score", 0)
            act = h.get("action", "adopted")
            strat = h.get("new_strategy", "")[:60]
            hist_lines.append(f"  [{act}] score={s:.1f}: {strat}...")
        hist_block = "\n".join(hist_lines) if hist_lines else "  (none)"
        return (
            "You are a gameplay strategy optimizer for a fantasy MMORPG agent.\n"
            f"Cycle: {cycle}\n\n"
            f"Over the last {self.eval_interval} cycles, performance was:\n"
            f"- Gold earned: {deltas['gold_delta']}\n"
            f"- XP gained: {deltas['xp_delta']}\n"
            f"- Mobs killed: {deltas['kills_delta']}\n"
            f"- Deaths: {deltas['deaths_delta']}\n"
            f"- Quests completed: {deltas['quests_delta']}\n"
            f"{ema_line}\n"
            f"Character state:\n{char_block}\n\n"
            f"Top tool usage: {tool_lines}\n\n"
            f"Recent journal:\n{journal_block}\n\n"
            f"Recent strategy history:\n{hist_block}\n\n"
            f"Current strategy:\n{current_strategy}\n\n"
            "Write an improved quest-chaining strategy (2-4 sentences). "
            "Focus on: which types of quests to prioritize, when to buy equipment/learn techniques, "
            "how to chain quests efficiently, and how to avoid deaths. "
            "Quest completions are the primary goal. "
            "Wrap your answer in <strategy></strategy> tags."
        )

    def _parse_strategy(self, response: str) -> str | None:
        m = re.search(r"<strategy>(.*?)</strategy>", response, re.DOTALL)
        if m:
            text = m.group(1).strip()
            if 10 < len(text) < 500:
                return text
        cleaned = response.strip()
        if 10 < len(cleaned) < 300:
            return cleaned
        return None

    def maybe_update(self, cycle, mem, model, tokenizer, tool_counts=None) -> str | None:
        if cycle % self.eval_interval != 0:
            return None
        current = self._take_snapshot(mem)
        if self._last_snapshot is None:
            self._last_snapshot = current
            return None
        print(f"[policy] Evaluating performance at cycle {cycle}...")
        deltas = self._compute_deltas(current, self._last_snapshot)
        current_strategy = get_current_strategy(mem)
        score = _compute_score(deltas["gold_delta"], deltas["quests_delta"], deltas["xp_delta"], deltas["deaths_delta"])
        if self._ema_score is None:
            self._ema_score = score
        else:
            self._prev_ema_score = self._ema_score
            self._ema_score = _EMA_ALPHA * score + (1 - _EMA_ALPHA) * self._ema_score
            if self._ema_score < self._prev_ema_score:
                self._consecutive_declines += 1
            else:
                self._consecutive_declines = 0
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
            print(f"[policy] Reverting to best strategy (score={score:.1f})")
            self._consecutive_declines = 0
        elif action in ("adopt", "explore"):
            meta_prompt = self._build_meta_prompt(deltas, current_strategy, mem, tool_counts, cycle)
            new_strategy = self._generate_new_strategy(meta_prompt, model, tokenizer)
            if new_strategy is None:
                self._last_snapshot = current
                self.last_action = "keep"
                self.last_ema_score = self._ema_score
                return None
        assert new_strategy is not None
        self._record_history(mem, cycle, current_strategy, new_strategy, deltas, score, action, self._ema_score)
        self._last_snapshot = current
        self.last_deltas = deltas
        self.last_improvement_score = score
        self.last_action = action
        self.last_ema_score = self._ema_score
        return new_strategy


def get_current_strategy(mem: dict) -> str:
    history = mem.get("policy_history", [])
    if history:
        return history[-1].get("new_strategy", DEFAULT_STRATEGY)
    return DEFAULT_STRATEGY
