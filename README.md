# WoG-Mistral-RL-0

**Autonomous MMORPG agent powered by Hermes-2-Pro-Mistral-7B running locally on Apple Silicon, with MCP tool calling and a self-improving RL policy loop.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![W&B](https://img.shields.io/badge/logging-Weights%20%26%20Biases-orange)

---

## Overview

WoG-Mistral-RL-0 is a 24/7 autonomous agent that plays [World of Gatherum](https://wog.gg), a fantasy MMORPG, entirely on its own. It runs **Hermes-2-Pro-Mistral-7B** (8-bit quantized) locally via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon, calls game actions through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), and continuously improves its gameplay strategy through a reinforcement-learning-style policy loop — all without any cloud LLM API calls.

The agent fights mobs, completes quests, gathers resources, crafts items, manages inventory, explores zones, and rewrites its own strategy based on measured performance deltas.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    WoG-Mistral-RL-0                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Ethereum  │───>│ Shard Auth   │───>│ MCP Session  │  │
│  │ Wallet    │    │ (JWT)        │    │ Auth         │  │
│  └──────────┘    └──────────────┘    └──────┬───────┘  │
│                                             │          │
│  ┌──────────────────────────────────────────▼───────┐  │
│  │              Autonomous Game Loop                 │  │
│  │  ┌────────────┐  ┌───────────┐  ┌─────────────┐  │  │
│  │  │ MLX Local  │─>│ Tool Call │─>│ MCP Server  │  │  │
│  │  │ Inference  │  │ Parser    │  │ (Game API)  │  │  │
│  │  │ (7B 8-bit) │  └───────────┘  └──────┬──────┘  │  │
│  │  └────────────┘                        │         │  │
│  │       ▲            ┌───────────────────▼──────┐  │  │
│  │       │            │ Response Truncation +    │  │  │
│  │       └────────────│ Memory Auto-Extraction   │  │  │
│  │                    └──────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Persistent     │  │ Policy      │  │ W&B         │  │
│  │ Memory         │  │ Evaluator   │  │ Telemetry   │  │
│  │ (.memory.json) │  │ (RL loop)   │  │ (metrics)   │  │
│  └────────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Local LLM inference** — Hermes-2-Pro-Mistral-7B (8-bit) via MLX, zero cloud API costs
- **MCP tool calling** — Hermes-2-Pro function calling format with `<tool_call>` XML tags, auto-injected session params
- **Ethereum wallet auth** — auto-generated, persisted to `.wallet_key`, signs shard challenges for JWT + MCP session auth
- **Persistent memory** — structured JSON store with facts, quests, zones, strategies, inventory notes, and journal (survives restarts)
- **Self-improving policy loop** — every 100 cycles, evaluates gold/XP/quest performance deltas and uses the LLM to rewrite its own strategy with EMA-smoothed scoring, epsilon-greedy exploration, and best-strategy rollback
- **W&B telemetry** — real-time dashboards for gameplay metrics, tool call distribution, inference timing, error rates, and policy evolution
- **Context management** — ChatML prompt format, history trimming (last 20 messages), JSON-aware response truncation (1500 chars)
- **Auto-recovery** — stuck detection with context reset after 3 consecutive no-tool cycles, error cooldown after 5 consecutive failures, automatic re-auth on session expiry

## Project Structure

```
mistral/
├── app.py             # Main agent: wallet, auth, MCP connection, game loop
├── memory.py          # Persistent memory system with auto-extraction from tool results
├── policy.py          # Self-improving RL policy evaluator (EMA scoring, meta-prompting)
├── wandb_logger.py    # W&B telemetry (gameplay, tools, policy, errors)
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores wallet key, memory file, wandb runs, pycache
├── .wallet_key        # (generated) Ethereum private key — gitignored
└── .memory.json       # (generated) Persistent agent memory — gitignored
```

## Prerequisites

- **Apple Silicon Mac** (M1/M2/M3/M4) — required for MLX inference
- **Python 3.11+**
- **WoG shard access** — the MCP URL and Shard URL are configured in `app.py`

## Quick Start

```bash
# Clone
git clone https://github.com/DasilvaKareem/WoG-Mistral-RL-0.git
cd WoG-Mistral-RL-0

# Install dependencies
pip install -r requirements.txt

# Run the agent
python app.py
```

On first run the agent will:
1. Generate an Ethereum wallet and save it to `.wallet_key`
2. Authenticate with the WoG shard (JWT) and MCP server
3. Create a character named "HermesAgent" (human warrior) if one doesn't exist
4. Deploy into the game world and start playing autonomously

No `.env` file is needed — the MCP and Shard URLs are configured directly in `app.py`.

## How It Works

### Authentication Flow

The agent uses a three-step auth process:

1. **Wallet creation** — generates an Ethereum keypair on first run, persists the private key to `.wallet_key` (chmod 600)
2. **Shard auth** — requests a challenge message from the shard, signs it with the wallet, and verifies the signature to receive a JWT
3. **MCP session auth** — after establishing a Streamable HTTP MCP connection, performs a second challenge/verify handshake through MCP tools (`auth_get_challenge` / `auth_verify_signature`) to authenticate the MCP session

### Game Loop

The core loop runs on a 3-second tick:

1. **Build prompt** — assembles a ChatML prompt with the system message (tools + strategy + memory) and conversation history
2. **LLM inference** — generates a response using MLX (`max_tokens=512`)
3. **Parse tool call** — extracts `<tool_call>` JSON from the model output (with fallback to bare JSON)
4. **Auto-inject params** — adds `sessionId`, `entityId`, and `zoneId` to the tool call args (hidden from the model to reduce prompt complexity)
5. **Execute via MCP** — calls the game tool through the MCP session
6. **Process result** — auto-extracts memory facts, truncates the response, and feeds it back as the next user message
7. **Policy check** — every 100 cycles, evaluates whether to keep, rewrite, explore, or revert the strategy

The model only sees ~30 core gameplay tools (filtered by `CORE_TOOL_PREFIXES`) out of the full MCP tool set, keeping the prompt focused for the 7B model.

### Memory System

Memory is stored in `.memory.json` with these sections:

| Section | Purpose | Limit |
|---------|---------|-------|
| `facts` | Character state (level, HP, gold, zone, gear) | 20 entries |
| `quests` | Active and completed quest tracking | — |
| `zones` | Explored zones with mob/resource/player counts | 15 zones |
| `strategies` | Agent-learned gameplay tips | 15 entries |
| `inventory_notes` | Notable items and crafting materials | 10 entries |
| `journal` | Timestamped event log | 30 entries |
| `stats` | Lifetime totals (kills, deaths, XP, gold, sessions) | — |
| `policy_history` | Strategy rewrite log with scores and deltas | 20 entries |

Memory is **auto-extracted** from tool results — `get_my_status` updates facts, `grind_mobs`/`fight_until_dead` update combat stats, `scan_zone` updates zone info, and quest tools track quest progress. The agent can also manually write memories via a local `remember` pseudo-tool.

### Policy Loop

Every 100 cycles the `PolicyEvaluator` runs:

1. **Snapshot** — captures current gold, XP, kills, deaths, quests completed
2. **Delta** — computes the difference from the previous snapshot
3. **Score** — weighted sum: `gold*3 + quests*50 + xp*0.1 + deaths*(-10)`
4. **EMA update** — exponential moving average (alpha=0.3) tracks performance trend
5. **Decision** — one of four actions:
   - **keep** (85%) — score is good and not declining
   - **explore** (15%) — epsilon-greedy random rewrite even when performing well
   - **adopt** — score is below threshold, generate a new strategy via meta-prompt
   - **revert** — 2+ consecutive declining windows, roll back to the best-ever strategy
6. **Meta-prompt** — if adopting/exploring, builds a rich prompt with performance deltas, character state, tool distribution, journal entries, and recent strategy history, then asks the LLM to write a new strategy in 2-4 sentences
7. **Apply** — updates memory, rewrites the system prompt, and logs to W&B

### Telemetry

All telemetry is logged to W&B under the `wog-agent` project with namespaced metrics:

| Namespace | Metrics |
|-----------|---------|
| `gameplay/` | total_kills, total_deaths, total_xp, total_gold_earned, gold_balance, quests_completed, quests_active |
| `system/` | inference_time_s, context_length, cycle, had_error, journal_entries |
| `tools/` | tool_name, total_calls, distribution bar chart (every 50 cycles) |
| `policy/` | improvement_score, ema_score, action, update_cycle, strategy history table |
| `errors/` | cycle, type, message |

W&B is optional — if `wandb` is not installed, all logging functions gracefully no-op.

## Configuration

Key constants in `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `mlx-community/Hermes-2-Pro-Mistral-7B-8bit` | MLX model to load |
| `TICK_INTERVAL` | `3` | Seconds between game loop cycles |
| `MAX_HISTORY` | `20` | Conversation messages to keep (plus system prompt) |
| `MAX_RESPONSE_CHARS` | `1500` | Max tool response length fed back to the model |
| `MAX_NO_TOOL_RETRIES` | `3` | No-tool cycles before context reset |

Key constants in `policy.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `eval_interval` | `100` | Cycles between policy evaluations |
| `_WEIGHT_GOLD` | `3` | Score weight for gold earned |
| `_WEIGHT_QUESTS` | `50` | Score weight per quest completed |
| `_WEIGHT_XP` | `0.1` | Score weight for XP gained |
| `_WEIGHT_DEATHS` | `-10` | Score penalty per death |
| `_ADOPT_THRESHOLD` | `30.0` | Score below which a new strategy is always tried |
| `_EPSILON` | `0.15` | Exploration rate when performing well |
| `_EMA_ALPHA` | `0.3` | EMA smoothing factor for score tracking |

## License

MIT
