# Design Document
# AlphaZero-Style Self-Play Training for 1v1 Catan

**Date:** 2026-08-04  
**Purpose:** Implementation guide for adding a self-play reinforcement learning system to an existing 1v1 Catan repository.

---

## Goal

Implement an AlphaZero-style training pipeline that:

- Uses the existing game engine.
- Learns entirely from self-play.
- Continuously improves over time.
- Maximizes simulation throughput.
- Supports future distributed scaling.
- Can replace an existing heuristic/search-based AI.

---

# 1. System Overview

```text
Existing Game Engine
        |
        v
Environment Interface
        |
        v
Self-Play Workers
        |
        v
Replay Buffer
        |
        v
Neural Network Trainer
        |
        v
Candidate Model Evaluation
        |
   Better than current?
      /         \
    Yes         No
     |           |
     v           |
Promote Model <--|
     |
     v
Continue Self-Play
```

This loop runs continuously.

---

# 2. Repository Refactor

Target structure:

```text
repo/
├── engine/
│   ├── board.py
│   ├── rules.py
│   ├── simulator.py
│   ├── actions.py
│   └── player.py
├── environment/
│   ├── catan_env.py
│   ├── state_encoder.py
│   └── action_encoder.py
├── models/
│   ├── network.py
│   └── checkpoints/
├── training/
│   ├── self_play.py
│   ├── replay_buffer.py
│   ├── trainer.py
│   └── evaluator.py
├── benchmark/
│   ├── benchmark.py
│   └── profiler.py
├── configs/
│   └── train.yaml
└── README.md
```

### Architectural Rules

- `engine/` must contain **no AI code**.
- `training/` must communicate only through the environment API.
- `models/` must be replaceable without changing engine code.

---

# 3. Environment API

Implement:

```python
class CatanEnv:
    def reset(self) -> State: ...
    def step(self, action: int) -> tuple[State, float, bool, dict]: ...
    def legal_actions(self) -> np.ndarray: ...
    def is_terminal(self) -> bool: ...
    def winner(self) -> int | None: ...
```

Requirements:

- Deterministic when seeded.
- No rendering during training.
- No logging during training.

---

# 4. State Encoding

Convert the full game state into tensors.

## Required Features

### Board

- Hex resource type (19)
- Hex probability number (19)
- Robber position (19)

### Vertices

- Empty / player1 settlement / player2 settlement
- Empty / player1 city / player2 city

### Edges

- Empty / player1 road / player2 road

### Global

- Current player
- Turn number
- Bank resources
- Remaining development cards

### Per Player

- Resource counts (5)
- Development card counts
- Victory points
- Longest road flag
- Largest army flag

Output should be a fixed-size tensor.

Avoid:

- Python objects
- Dictionaries
- Strings

Use contiguous NumPy arrays or PyTorch tensors.

---

# 5. Action Encoding

Map every legal action to a fixed integer.

Example:

```text
0   End turn
1   Build road edge 0
2   Build road edge 1
...
73  Build settlement vertex 0
...
150 Build city vertex 0
...
```

Implement:

```python
action_mask = env.legal_actions()
```

Where `action_mask[i] == 1` if legal.

Illegal actions must be masked before sampling.

---

# 6. Performance Benchmarking

Create `benchmark/benchmark.py`.

Measure:

- Games played
- Elapsed time
- Games/sec
- Turns/sec
- Average turns/game
- Average ms/game
- CPU utilization
- Memory utilization

Example output:

```text
Games: 10000
Elapsed: 48.2 s
Games/sec: 207.4
Average game: 4.82 ms
Average turns: 113
```

This benchmark is the primary optimization metric.

---

# 7. Performance Targets

| Stage | Target ms/game |
|---|---:|
| Initial | <100 ms |
| Good | <20 ms |
| Excellent | <5 ms |
| Research-grade | <1 ms |

Training becomes practical below ~10 ms/game.

---

# 8. Mandatory Optimizations

Disable during training:

- Rendering
- Animations
- Printing
- Logging
- JSON export
- Network calls
- Sleep calls

Profile:

- Deep copies
- Legal move generation
- Resource distribution
- Longest road calculation
- Memory allocations

Prefer in-place mutation with undo stacks over full state copies.

---

# 9. Parallel Self-Play

Use multiprocessing.

```python
from multiprocessing import Pool

with Pool(processes=16) as pool:
    results = pool.map(play_game, range(10000))
```

Expected scaling:

| Cores | Games/sec |
|---|---:|
| 1 | 100 |
| 8 | ~700 |
| 16 | ~1400 |
| 32 | ~2600 |

CPU workers generate games while GPU trains.

---

# 10. Neural Network

## Recommended

Graph Neural Network (GNN)

Reason: Catan is naturally a graph.

Alternative: Transformer.

## Outputs

### Policy Head

Probability distribution over all actions.

### Value Head

Estimated probability of winning in [-1, 1].

---

# 11. Monte Carlo Tree Search

Tree nodes:

- Player nodes
- Chance nodes (dice)

At chance nodes, sample according to dice probabilities:

| Roll | Probability |
|---|---:|
| 2 | 1/36 |
| 3 | 2/36 |
| 4 | 3/36 |
| 5 | 4/36 |
| 6 | 5/36 |
| 7 | 6/36 |
| 8 | 5/36 |
| 9 | 4/36 |
| 10 | 3/36 |
| 11 | 2/36 |
| 12 | 1/36 |

Use ~100–400 simulations per move initially.

---

# 12. Replay Buffer

Store tuples:

```python
(state, policy_target, value_target)
```

Recommended size:

```text
2,000,000 positions
```

Sampling strategy:

- 25% newest
- 25% recent
- 25% medium
- 25% old

---

# 13. Continuous Training Loop

Do **not** wait for millions of games before training.

Correct loop:

```text
Generate 20,000 games
Train 1,000 batches
Generate 20,000 games
Train 1,000 batches
Repeat forever
```

---

# 14. Evaluation

Run head-to-head matches:

```text
CurrentBest vs Candidate
```

Recommended:

- 1,000 games
- Alternate starting player
- Multiple random board seeds

Promote if:

```text
Win rate > 55%
```

Otherwise discard.

---

# 15. Exploration Schedule

Training:

```text
Temperature = 1.0 for first 20 turns
Temperature = 0.1 afterwards
```

Evaluation:

```text
Temperature = 0.0
```

---

# 16. Measuring Required Games

Let:

```text
t = average ms/game
c = number of CPU cores
```

Games/sec:

```text
gps = c * 1000 / t
```

Example:

```text
t = 7.3 ms
c = 24
gps ≈ 3288
```

Time for 1 million games:

```text
1,000,000 / 3288 ≈ 304 s ≈ 5.1 min
```

Use this formula before large training runs.

---

# 17. Choosing Training Frequency

Suppose:

```text
Replay update size = 20,000 games
gps = 5,000 games/sec
```

Generation time:

```text
20,000 / 5,000 = 4 s
```

Therefore trigger training every ~4 seconds.

If training is slower than generation:

- Increase GPU count
- Reduce batch count
- Increase replay size

---

# 18. Metrics Dashboard

Log continuously:

## Simulation

- games/sec
- turns/sec
- average game length
- average turn latency

## Training

- policy loss
- value loss
- entropy
- learning rate
- gradient norm

## Evaluation

- win rate vs best model
- ELO estimate
- opening diversity

## System

- GPU utilization
- CPU utilization
- replay fill level

---

# 19. Scaling Roadmap

## Stage 1

Single machine:

```text
CPU -> self-play
GPU -> training
```

## Stage 2

Many CPU workers + one GPU.

## Stage 3

Multiple machines:

```text
Workers -> Central Replay Buffer -> GPU Trainers
```

---

# 20. Development Milestones

## Milestone 1 — Benchmark

- Add benchmark.
- Measure baseline.
- Remove rendering/logging.
- Optimize hotspots.

## Milestone 2 — Environment

- Implement RL API.
- Add state encoder.
- Add action encoder.

## Milestone 3 — Random Self-Play

- Generate random games.
- Verify replay integrity.
- Verify determinism with seeds.

## Milestone 4 — Neural Network

- Train policy/value network.
- Monitor losses.

## Milestone 5 — MCTS

- Integrate search.
- Tune simulations.

## Milestone 6 — Continuous Learning

- Run indefinitely.
- Evaluate periodically.
- Promote stronger checkpoints.

---

# 21. Validation Checklist

Before long training runs:

- [ ] Deterministic with fixed seed
- [ ] No memory leaks
- [ ] No rendering during training
- [ ] Legal action mask correct
- [ ] State tensor fixed shape
- [ ] Action IDs stable
- [ ] Benchmark reproducible
- [ ] Evaluation reproducible
- [ ] Checkpoint save/load works
- [ ] Self-play parallelism stable

---

# 22. Recommended Initial Configuration

```yaml
self_play_workers: 16
replay_buffer_size: 2000000
games_per_iteration: 20000
batch_size: 512
training_batches: 1000
learning_rate: 0.001
mcts_simulations: 200
evaluation_games: 1000
promotion_threshold: 0.55
temperature_opening_turns: 20
checkpoint_interval_minutes: 30
```

---

# 23. Expected Timeline

Assuming ~5 ms/game and 16 cores:

| Milestone | Estimate |
|---|---:|
| Benchmarking | 1–3 days |
| Environment API | 1–2 days |
| State/action encoding | 2–4 days |
| Parallel self-play | 1–2 days |
| Replay buffer | 1 day |
| Network implementation | 2–5 days |
| MCTS integration | 3–7 days |
| Evaluation pipeline | 1–2 days |
| First learning agent | ~2 weeks |

---

# 24. Final Engineering Priorities

Order of importance:

1. Simulation speed
2. Deterministic environment
3. Correct legal action masking
4. Clean state encoding
5. Parallel self-play
6. Stable replay buffer
7. Evaluation pipeline
8. Neural network sophistication

A fast, deterministic simulator with continuous self-play will usually outperform a more sophisticated model trained on a slow environment.
