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

try:
    import torch as _torch
    _torch_available = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _torch_available = False

_run = None
_tool_counts: Counter = Counter()
_zone_counts: Counter = Counter()


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
            settings=wandb.Settings(x_stats_sampling_interval=10),
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

    # Track zone occupancy
    current_zone = facts.get("zone") or "unknown"
    _zone_counts[current_zone] += 1

    metrics = {
        # Gameplay
        "gameplay/total_kills": stats.get("total_kills", 0),
        "gameplay/total_deaths": stats.get("total_deaths", 0),
        "gameplay/total_xp": stats.get("total_xp", 0),
        "gameplay/total_gold_earned": stats.get("total_gold_earned", 0),
        "gameplay/gold_balance": _parse_gold(facts.get("gold")),
        "gameplay/quests_completed": len(mem.get("quests", {}).get("completed", [])),
        "gameplay/quests_active": len(mem.get("quests", {}).get("active", [])),
        "gameplay/quests_available": len(mem.get("quests", {}).get("available", [])),
        "gameplay/quest_xp": stats.get("total_quests_xp", 0),
        "gameplay/quest_gold": stats.get("total_quests_gold", 0),
        # Quest timing
        "quests/avg_completion_time_s": _avg(stats.get("quest_completion_times", [])),
        "quests/fastest_completion_s": min(stats.get("quest_completion_times") or [0]) or 0,
        "quests/total_timed_completions": len(stats.get("quest_completion_times", [])),
        # Movement
        "movement/current_zone": current_zone,
        "movement/total_transitions": stats.get("total_zone_transitions", 0),
        "movement/zones_discovered": len(stats.get("zone_visit_counts", {})),
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

    # Log per-difficulty quest metrics
    by_diff = stats.get("quests_by_difficulty", {})
    for diff, data in by_diff.items():
        avg_t = _avg(data.get("times", []))
        try:
            _run.log({
                f"quests_by_difficulty/{diff}/count": data.get("count", 0),
                f"quests_by_difficulty/{diff}/total_xp": data.get("total_xp", 0),
                f"quests_by_difficulty/{diff}/total_gold": data.get("total_gold", 0),
                f"quests_by_difficulty/{diff}/avg_time_s": avg_t,
            }, step=cycle)
        except Exception:
            pass

    # Log distribution charts every 50 cycles
    if cycle % 50 == 0:
        if _tool_counts:
            _log_tool_distribution(cycle)
        if _zone_counts:
            _log_zone_distribution(cycle)
        if by_diff:
            _log_quest_difficulty_distribution(cycle, by_diff)


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


def _log_zone_distribution(cycle: int) -> None:
    """Log zone time distribution as a W&B bar chart."""
    if _run is None:
        return
    try:
        table = wandb.Table(columns=["zone", "cycles_spent"], data=[
            [zone, count] for zone, count in _zone_counts.most_common(20)
        ])
        _run.log({
            "movement/zone_distribution": wandb.plot.bar(
                table, "zone", "cycles_spent", title=f"Zone occupancy (cycle {cycle})"
            ),
        }, step=cycle)
    except Exception:
        pass


def _log_quest_difficulty_distribution(cycle: int, by_diff: dict) -> None:
    """Log quest count and avg time per difficulty as W&B bar charts."""
    if _run is None:
        return
    try:
        # Quest count by difficulty
        count_table = wandb.Table(
            columns=["difficulty", "quests_completed"],
            data=[[d, data.get("count", 0)] for d, data in sorted(by_diff.items())],
        )
        _run.log({
            "quests/difficulty_counts": wandb.plot.bar(
                count_table, "difficulty", "quests_completed",
                title=f"Quests by Difficulty (cycle {cycle})",
            ),
        }, step=cycle)

        # Avg time by difficulty
        time_data = [
            [d, _avg(data.get("times", []))]
            for d, data in sorted(by_diff.items())
            if data.get("times")
        ]
        if time_data:
            time_table = wandb.Table(columns=["difficulty", "avg_time_s"], data=time_data)
            _run.log({
                "quests/difficulty_avg_time": wandb.plot.bar(
                    time_table, "difficulty", "avg_time_s",
                    title=f"Avg Quest Time by Difficulty (cycle {cycle})",
                ),
            }, step=cycle)

        # XP per difficulty
        xp_table = wandb.Table(
            columns=["difficulty", "total_xp"],
            data=[[d, data.get("total_xp", 0)] for d, data in sorted(by_diff.items())],
        )
        _run.log({
            "quests/difficulty_xp": wandb.plot.bar(
                xp_table, "difficulty", "total_xp",
                title=f"XP by Difficulty (cycle {cycle})",
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
    ema_score: float = 0.0,
    action: str = "adopted",
) -> None:
    """Log a policy rewrite event with old/new strategy text and perf deltas."""
    if _run is None:
        return
    try:
        _run.log({
            "policy/improvement_score": improvement_score,
            "policy/update_cycle": cycle,
            "policy/ema_score": ema_score,
            "policy/action": action,
        }, step=cycle)

        table = wandb.Table(columns=[
            "cycle", "old_strategy", "new_strategy",
            "gold_delta", "xp_delta", "kills_delta", "deaths_delta",
            "quests_delta", "zones_discovered_delta", "zone_transitions_delta",
            "improvement_score", "ema_score", "action",
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
            performance_snapshot.get("zones_discovered_delta", 0),
            performance_snapshot.get("zone_transitions_delta", 0),
            improvement_score,
            ema_score,
            action,
        )
        _run.log({"policy/history": table}, step=cycle)
    except Exception:
        pass


def log_gpu_stats(cycle: int) -> None:
    """Log CUDA GPU memory stats if available."""
    if _run is None or not _torch_available or _torch is None:
        return
    try:
        if not _torch.cuda.is_available():
            return
        metrics = {}
        for i in range(_torch.cuda.device_count()):
            alloc = _torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = _torch.cuda.memory_reserved(i) / 1024 ** 3
            total = _torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            metrics[f"gpu/{i}/memory_allocated_gb"] = alloc
            metrics[f"gpu/{i}/memory_reserved_gb"] = reserved
            metrics[f"gpu/{i}/memory_utilization_pct"] = (alloc / total * 100) if total else 0
        _run.log(metrics, step=cycle)
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
            "quest_xp": stats.get("total_quests_xp", 0),
            "quest_gold": stats.get("total_quests_gold", 0),
            "zone_transitions": stats.get("total_zone_transitions", 0),
            "zones_discovered": len(stats.get("zone_visit_counts", {})),
            "zone_visit_counts": stats.get("zone_visit_counts", {}),
            "avg_quest_completion_time_s": _avg(stats.get("quest_completion_times", [])),
            "quest_completion_times": stats.get("quest_completion_times", []),
            "sessions": stats.get("sessions", 0),
            "tool_calls": dict(_tool_counts),
            "zone_time": dict(_zone_counts),
        })
        _run.finish()
        print("[wandb] Run finished.")
    except Exception as e:
        print(f"[wandb] Error finishing run: {e}")


def _avg(lst: list) -> float:
    """Safe average of a list of numbers."""
    return sum(lst) / len(lst) if lst else 0.0


def _parse_gold(val) -> float:
    """Safely parse gold value from memory facts."""
    if val is None or val == "?":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
