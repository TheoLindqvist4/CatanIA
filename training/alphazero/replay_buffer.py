"""Where self-play positions live between being generated and being learned from.

The guide asks for two million positions and for sampling stratified by age. Two million is
not affordable here and the reason is arithmetic rather than taste: an observation is 1,884
floats, so two million of them is 15 GB in float32 on a machine that also has to hold twenty
worker processes. Three things bring it into range.

**Observations are stored as float16.** Halves the bill. The encoder emits counts, ratios and
flags — nothing outside float16's exact-integer range of 2048 and nothing that needs more
than three significant digits, which ``tests/test_alphazero.py`` checks by round-tripping a
real observation.

**Policy targets are stored sparsely.** A visit-count distribution over 325 actions after 48
simulations has at most 48 non-zero entries and usually under a dozen. Keeping the top
:data:`POLICY_TOP_K` and renormalising costs 128 bytes instead of 1,300, and drops only
entries a single visit apart from zero.

**The buffer is sized to the run.** The default holds :data:`DEFAULT_CAPACITY` positions,
about 850 MB, which is several times what a three-hour run on twenty cores produces — so for
that run the ring never wraps and "stratified by age" spans the whole history.

Sampling follows the guide: the buffer is cut into four equal age bands and a batch is drawn
in equal parts from each. Uniform sampling over a ring that is being appended to at speed is
biased toward the newest data exactly when the newest data is the most correlated — every
position in a game shares one outcome, so a batch drawn from one iteration is a batch of a
few dozen games seen 300 times. Banding by age is the cheap fix, and unlike a priority scheme
it needs no bookkeeping per sample.
"""

import numpy as np

from catan import action_space, encoder

#: How many actions of a policy target are kept. Above the number of simulations any run
#: here can afford, so nothing with a visit is ever dropped.
POLICY_TOP_K = 48

#: Positions held by default. At 1,884 float16 observations that is about 850 MB.
DEFAULT_CAPACITY = 220_000

#: Age bands a batch is drawn from in equal parts. The guide's 25/25/25/25.
AGE_BANDS = 4

_MASK_BYTES = (action_space.NUM_ACTIONS + 7) // 8


def pack_mask(mask):
    """A legality mask as :data:`_MASK_BYTES` bytes.

    Takes either the engine's ``bytearray`` or a bool array, because both exist in this
    package: the environment hands out the former and the search builds the latter.
    """
    if isinstance(mask, np.ndarray):
        return np.packbits(mask.astype(bool, copy=False))
    return np.packbits(np.frombuffer(bytes(mask), dtype=np.uint8).astype(bool))


def unpack_masks(packed):
    """``(rows, NUM_ACTIONS)`` bools from packed rows."""
    return np.unpackbits(packed, axis=1, count=action_space.NUM_ACTIONS).astype(bool)


#: Padding slot for a sparse policy row. One past the action space, so densifying can scatter
#: every slot unconditionally and then drop this column. Padding with a *real* index instead
#: — 0, the obvious choice — silently competes with END_TURN for the same cell.
PAD_INDEX = action_space.NUM_ACTIONS


def sparse_policy(actions, counts):
    """A visit-count distribution as ``(indices, probabilities)`` of length :data:`POLICY_TOP_K`.

    Unused slots are :data:`PAD_INDEX` at probability 0, so no separate length is stored.
    """
    indices = np.full(POLICY_TOP_K, PAD_INDEX, dtype=np.int16)
    probabilities = np.zeros(POLICY_TOP_K, dtype=np.float16)
    total = counts.sum()
    if total <= 0:
        return indices, probabilities
    keep = np.argsort(counts)[::-1][:POLICY_TOP_K]
    keep = keep[counts[keep] > 0]
    indices[:len(keep)] = actions[keep]
    probabilities[:len(keep)] = (counts[keep] / counts[keep].sum()).astype(np.float16)
    return indices, probabilities


class ReplayBuffer:
    """A fixed-capacity ring of ``(observation, policy target, value target)``.

    Args:
        capacity: positions held before the oldest are overwritten.
    """

    def __init__(self, capacity=DEFAULT_CAPACITY):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, encoder.SIZE), dtype=np.float16)
        self.policy_index = np.zeros((self.capacity, POLICY_TOP_K), dtype=np.int16)
        self.policy_prob = np.zeros((self.capacity, POLICY_TOP_K), dtype=np.float16)
        self.mask = np.zeros((self.capacity, _MASK_BYTES), dtype=np.uint8)
        self.value = np.zeros(self.capacity, dtype=np.int8)
        self.cursor = 0
        self.size = 0
        #: Positions ever written. Age is derived from this rather than stored per row.
        self.written = 0

    def __len__(self):
        return self.size

    @property
    def full(self):
        return self.size >= self.capacity

    def add(self, obs, policy_index, policy_prob, mask, value):
        """Append one batch of positions, wrapping when full.

        Every argument is an array whose first axis is the batch. Written in at most two
        slices rather than row by row: a self-play iteration delivers tens of thousands of
        positions and a Python loop over them measured as 40% of the trainer's own time.
        """
        count = len(obs)
        if count == 0:
            return 0
        if count >= self.capacity:
            # More than fits. Keep the newest, which is what a ring would have left anyway.
            obs, policy_index, policy_prob, mask, value = (
                a[-self.capacity:] for a in (obs, policy_index, policy_prob, mask, value)
            )
            count = self.capacity

        first = min(count, self.capacity - self.cursor)
        for target, source in (
            (self.obs, obs), (self.policy_index, policy_index),
            (self.policy_prob, policy_prob), (self.mask, mask), (self.value, value),
        ):
            target[self.cursor:self.cursor + first] = source[:first]
            if count > first:
                target[:count - first] = source[first:]

        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.written += count
        return count

    # ------------------------------------------------------------------ #

    def _ordered(self):
        """Row indices oldest-first.

        Before the ring wraps, that is simply ``0..size``. After, the oldest row is the one
        the cursor is about to overwrite.
        """
        if not self.full:
            return np.arange(self.size)
        return np.concatenate([np.arange(self.cursor, self.capacity),
                               np.arange(0, self.cursor)])

    def sample(self, batch_size, rng, bands=AGE_BANDS):
        """Draw a batch in equal parts from ``bands`` equal age bands, newest band last.

        Returns ``(obs, policy, mask, value)`` as float32/float32/bool/float32 numpy arrays,
        with the policy densified to the full action space.
        """
        if self.size == 0:
            raise ValueError("the buffer is empty")
        order = self._ordered()
        bands = max(1, min(bands, self.size))
        edges = np.linspace(0, self.size, bands + 1).astype(int)

        share = batch_size // bands
        picks = []
        for band in range(bands):
            lo, hi = edges[band], edges[band + 1]
            wanted = share + (batch_size - share * bands if band == bands - 1 else 0)
            if hi <= lo or wanted <= 0:
                continue
            picks.append(order[rng.integers(lo, hi, size=wanted)])
        rows = np.concatenate(picks)

        obs = self.obs[rows].astype(np.float32)
        mask = unpack_masks(self.mask[rows])
        value = self.value[rows].astype(np.float32)

        # One column wider than the action space; the extra one collects every padded slot
        # and is then dropped.
        policy = np.zeros((len(rows), action_space.NUM_ACTIONS + 1), dtype=np.float32)
        np.put_along_axis(policy, self.policy_index[rows].astype(np.int64),
                          self.policy_prob[rows].astype(np.float32), axis=1)
        return obs, policy[:, :action_space.NUM_ACTIONS], mask, value

    # ------------------------------------------------------------------ #

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "fill": self.size / self.capacity,
            "written": self.written,
            "wrapped": bool(self.full and self.written > self.capacity),
        }

    def __repr__(self):
        return (f"ReplayBuffer({self.size:,}/{self.capacity:,} positions, "
                f"{self.written:,} written)")
