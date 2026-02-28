"""
W&B telemetry for the WoG agent.

Logs gameplay metrics, tool call distribution, inference timing,
error rates, and policy update events. All functions gracefully
no-op if wandb is not initialized.
"""

from collections import Counter

try:
    import wandb
except ImportError:
    wandb = None

_run = None
_tool_counts: Counter = Counter()


def init_run(config: dict) -> None:
    """Start a W&B run with project 'wog-agent' and agent config."""
    global _run
    if wandb is None:
        print("[wandb] wandb not installed, skipping telemetry")
        return
    try:
        _run = wandb.init(
            project="wog-agent",
            config=config,
            reinit=True,
        )
        print(f"[wandb] Run started: {_run.url}")
    except Exception as e:
        print(f"[wandb] Failed to init: {e}")
        _run = None


def log_cycle(
    cycle: int,
    tool_name: str | None,
    tool_args: dict | None,
    result_preview: str,
    mem: dict,
    inference_time_s: float,
    context_length: int,
    had_error: bool,
) -> None:
    """Log per-cycle metrics under gameplay/, system/, tools/ namespaces."""
    if _run is None:
        return

    stats = mem.get("stats", {})
    facts = mem.get("facts", {})

    # Track tool distribution
    if tool_name:
        _tool_counts[tool_name] += 1

    metrics = {
        # Gameplay
        "gameplay/total_kills": stats.get("total_kills", 0),
        "gameplay/total_deaths": stats.get("total_deaths", 0),
        "gameplay/total_xp": stats.get("total_xp", 0),
        "gameplay/total_gold_earned": stats.get("total_gold_earned", 0),
        "gameplay/gold_balance": _parse_gold(facts.get("gold")),
        "gameplay/quests_completed": len(mem.get("quests", {}).get("completed", [])),
        "gameplay/quests_active": len(mem.get("quests", {}).get("active", [])),
        # System
        "system/inference_time_s": inference_time_s,
        "system/context_length": context_length,
        "system/cycle": cycle,
        "system/had_error": int(had_error),
        "system/journal_entries": len(mem.get("journal", [])),
        # Tools
        "tools/tool_name": tool_name or "none",
        "tools/total_calls": sum(_tool_counts.values()),
    }

    try:
        _run.log(metrics, step=cycle)
    except Exception:
        pass

    # Log tool distribution bar chart every 50 cycles
    if cycle % 50 == 0 and _tool_counts:
        _log_tool_distribution(cycle)


def _log_tool_distribution(cycle: int) -> None:
    """Log tool call distribution as a W&B bar chart."""
    if _run is None:
        return
    try:
        table = wandb.Table(columns=["tool", "count"], data=[
            [name, count] for name, count in _tool_counts.most_common(20)
        ])
        _run.log({
            "tools/distribution": wandb.plot.bar(
                table, "tool", "count", title=f"Tool calls (cycle {cycle})"
            ),
        }, step=cycle)
    except Exception:
        pass


def log_policy_update(
    cycle: int,
    old_strategy: str,
    new_strategy: str,
    performance_snapshot: dict,
    improvement_score: float,
) -> None:
    """Log a policy rewrite event with old/new strategy text and perf deltas."""
    if _run is None:
        return
    try:
        _run.log({
            "policy/improvement_score": improvement_score,
            "policy/update_cycle": cycle,
        }, step=cycle)

        table = wandb.Table(columns=[
            "cycle", "old_strategy", "new_strategy",
            "gold_delta", "xp_delta", "kills_delta", "deaths_delta",
            "quests_delta", "improvement_score",
        ])
        table.add_data(
            cycle,
            old_strategy[:500],
            new_strategy[:500],
            performance_snapshot.get("gold_delta", 0),
            performance_snapshot.get("xp_delta", 0),
            performance_snapshot.get("kills_delta", 0),
            performance_snapshot.get("deaths_delta", 0),
            performance_snapshot.get("quests_delta", 0),
            improvement_score,
        )
        _run.log({"policy/history": table}, step=cycle)
    except Exception:
        pass


def log_error(cycle: int, error_type: str, error_msg: str) -> None:
    """Log errors under errors/ namespace."""
    if _run is None:
        return
    try:
        _run.log({
            "errors/cycle": cycle,
            "errors/type": error_type,
            "errors/message": error_msg[:300],
        }, step=cycle)
    except Exception:
        pass


def finish(mem: dict) -> None:
    """Write summary stats and finish the W&B run."""
    if _run is None:
        return
    try:
        stats = mem.get("stats", {})
        _run.summary.update({
            "total_kills": stats.get("total_kills", 0),
            "total_deaths": stats.get("total_deaths", 0),
            "total_xp": stats.get("total_xp", 0),
            "total_gold_earned": stats.get("total_gold_earned", 0),
            "quests_completed": len(mem.get("quests", {}).get("completed", [])),
            "sessions": stats.get("sessions", 0),
            "tool_calls": dict(_tool_counts),
        })
        _run.finish()
        print("[wandb] Run finished.")
    except Exception as e:
        print(f"[wandb] Error finishing run: {e}")


def _parse_gold(val) -> float:
    """Safely parse gold value from memory facts."""
    if val is None or val == "?":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
