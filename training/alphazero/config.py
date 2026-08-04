"""The run's settings, and where they are read from.

The guide specifies ``configs/train.yaml``, and that is what this reads — but PyYAML is not
a dependency of this project and adding one for a file of twenty ``key: value`` lines is a
poor trade. So :func:`load_config` parses the subset the guide's example actually uses: flat
scalars, ``#`` comments, and nothing nested. A file that uses more than that is **rejected
with the offending line**, rather than half-parsed into a config that is quietly wrong — a
training run misconfigured by a silently dropped key is a wasted afternoon that looks like a
bad idea.

Defaults live in :data:`DEFAULTS`, so a missing key is never a crash and the YAML is a set of
overrides rather than a required document.
"""

import pathlib

#: Everything the pipeline reads, with the value used when the config does not say.
#:
#: The numbers that differ from the guide's recommended configuration are marked, because
#: each is a deliberate response to *this* machine — a 20-core CPU with no GPU — rather than
#: a preference. See ``docs/decisions/0023-alphazero-self-play.md``.
DEFAULTS = {
    # --- self-play ------------------------------------------------------------------- #
    "self_play_workers": 14,        # guide: 16. Leaves cores for the trainer and the web UI.
    "envs_per_worker": 12,          # games in flight per worker; sets the evaluation batch.
                                    # Games in a worker advance in lockstep, one simulation
                                    # each per round, so they also *finish* together: the
                                    # width is how long the pipeline takes to deliver its
                                    # first samples. At 24 that is ~65s of worker time and
                                    # samples then arrive in one lump; at 12 it is half that,
                                    # for about 6% more time per network evaluation.
    "mcts_simulations": 96,         # guide: 200. Not 200 because that is ~4 games/sec across
                                    # the whole machine; not 48 because at 48 the labels are
                                    # only four points better than the policy already is.
                                    # See configs/train.yaml for the measurement.
    "max_turns": 400,
    "temperature": 1.0,
    "temperature_final": 0.15,
    "temperature_opening_turns": 20,
    "c_puct": 1.5,
    "fpu": 0.25,
    "dirichlet_alpha": 0.5,
    "dirichlet_weight": 0.10,  # AlphaZero uses 0.25 at 800 sims; at 96 it flips 24% of labels

    # --- replay ---------------------------------------------------------------------- #
    "replay_buffer_size": 180_000,  # guide: 2,000,000. The observation is 2,503 floats,
                                    # so this is ~0.9 GB in float16; see replay_buffer.
    "min_buffer": 4_000,            # positions before the first gradient step

    # --- learning -------------------------------------------------------------------- #
    "batch_size": 512,
    "training_batches": 60,        # guide: 1,000, paired with 20,000 games per iteration
    "generate_seconds": 35,         # per iteration. A clock, not a sample count: see
                                    # Generator.run for why a count wastes most of the pool.
    "positions_per_iteration": 0,   # 0 means "use the clock". Kept for benchmarks and tests.
    "learning_rate": 7e-5,          # guide: 1e-3. Warm-started, so the low end. 2e-4 was
                                    # above the 3e-5..1e-4 band CLAUDE.md gives for
                                    # fine-tuning, and a run at 2e-4 peaked and then drifted
                                    # back down over the next 40 iterations.
    "weight_decay": 1e-4,
    "value_weight": 1.0,
    "grad_clip": 1.0,

    # --- evaluation and promotion ---------------------------------------------------- #
    #
    # The in-loop check is deliberately cheap and deliberately measures the *network*, not
    # the searcher. `play_match` is sequential and a searching agent evaluates the network
    # once per simulation at batch 1, so 200 games at 32 simulations costs about thirteen
    # minutes — which, every six iterations, is most of a training run. At 0 simulations the
    # agent plays its raw policy, which is one forward pass a move and takes about 40
    # seconds; that still answers the only question the loop needs answered, which is whether
    # the network is getting better.
    #
    # Whether the *search* helps is a different question, and it is answered once, properly,
    # by the promotion gate at `promotion_simulations`.
    "evaluation_games": 100,        # guide: 1,000. Wilson interval decides, not the count.
    "promotion_games": 400,
    "promotion_threshold": 0.55,
    "evaluate_every": 20,           # iterations
    "eval_simulations": 0,          # 0 = the raw policy; see above

    # --- housekeeping ---------------------------------------------------------------- #
    "checkpoint_interval_minutes": 15,
    "seed": 0,
    "warm_start": "models/champion_az.pt",
    "run_directory": "checkpoints/alphazero",
}

CONFIG_PATH = pathlib.Path("configs/train.yaml")


class Config(dict):
    """A dict that refuses unknown keys, so a typo is not a silently ignored setting."""

    def __init__(self, values=None):
        super().__init__(DEFAULTS)
        if values:
            unknown = set(values) - set(DEFAULTS)
            if unknown:
                raise KeyError(
                    f"unknown setting(s) {sorted(unknown)} — known settings are "
                    f"{sorted(DEFAULTS)}"
                )
            self.update(values)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def describe(self):
        """The settings that differ from the defaults, for a run's header line."""
        return {k: v for k, v in self.items() if DEFAULTS[k] != v}


def parse(text):
    """The flat subset of YAML the guide's configuration uses.

    Raises:
        ValueError: on nesting, lists, or anything else outside ``key: scalar``.
    """
    values = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line[0] in " \t-":
            raise ValueError(
                f"{CONFIG_PATH}:{number}: only flat 'key: value' lines are supported, "
                f"got {raw.strip()!r}"
            )
        if ":" not in line:
            raise ValueError(f"{CONFIG_PATH}:{number}: expected 'key: value', got {raw!r}")
        key, _, value = line.partition(":")
        values[key.strip()] = _scalar(value.strip())
    return values


def _scalar(text):
    if text in ("true", "True", "yes"):
        return True
    if text in ("false", "False", "no"):
        return False
    if text in ("null", "~", ""):
        return None
    body = text.replace("_", "")
    try:
        return int(body)
    except ValueError:
        pass
    try:
        return float(body)
    except ValueError:
        pass
    return text.strip("'\"")


def load_config(path=CONFIG_PATH, **overrides):
    """The configuration, from ``path`` if it exists, with keyword overrides on top."""
    path = pathlib.Path(path)
    values = parse(path.read_text(encoding="utf-8")) if path.is_file() else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(values)
