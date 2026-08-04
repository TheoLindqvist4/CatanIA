"""PUCT search over a determinized Catan position.

Textbook AlphaZero assumes a two-player, perfect-information, deterministic game. Catan is
none of those. Three departures, each of which is a decision rather than an oversight.

**Chance nodes.** The dice are not a move. A node whose state can only be resolved by rolling
is a :data:`CHANCE` node, and descending through it *samples* a roll from the state's own
generator — which, after determinization, is a fresh shuffle of the balanced deck rather than
the deck the real game will actually deal. Children are keyed by the roll **total**, not by
the pair, because 5+2 and 3+4 lead to the same game. Revisiting a chance node re-samples: the
outcome distribution is therefore the game's, and a child's value is visited in proportion to
how often that total comes up. That is the correct expectation in the limit, and it is much
cheaper than enumerating all eleven totals at every roll.

**Hidden information.** The tree is built on a state from :mod:`training.alphazero.determinize`
— one world drawn from the searching player's information set — never on the true state. See
that module for why this is the boundary the whole package is built around. One particle per
search rather than an average over many: at the simulation counts that fit on a CPU, spending
the budget on depth in one consistent world measures better than spreading it thinly over
several, and the *policy target* is averaged across games and positions anyway.

**Not strictly alternating.** During a discard the decision belongs to whoever is over the
hand limit, which may be either player, and after a 7 the roller acts twice in a row. Sign
flipping by depth would therefore be wrong. Every node records **which seat acts**, and values
are propagated in a fixed frame — as seat 1's value — with each node reading off its own
perspective. This is why ``num_players == 2`` is checked: the fixed frame is what makes a
zero-sum value scalar meaningful at all.

**Forced moves are collapsed.** Catan is full of positions with exactly one legal action —
setup roads, single-resource discards, a robber move with one destination. Searching them
costs a network evaluation to choose between one option. Descent applies them immediately and
keeps going, so the budget is spent only on real decisions. Measured at roughly a third of all
states.

The search does not own the network. :meth:`Search.request` hands back a position to evaluate
and :meth:`Search.deliver` takes the answer, so a caller can run many searches at once and
evaluate all their leaves in one batch — which is worth about 7x on this machine, and is the
single largest throughput lever in the package.
"""

import math

import numpy as np

from catan import action_space, encoder, rules
from catan.state import Phase

#: Node kinds. A decision node has legal actions; a chance node is waiting on the dice.
DECISION, CHANCE, TERMINAL = 0, 1, 2

#: Exploration constant in the PUCT rule. 1.5 rather than AlphaGo's 5-ish because the branching
#: factor here is small (a typical Catan position offers 5-30 moves, not 250) and the prior is
#: warm-started, so leaning on it harder is the right trade.
C_PUCT = 1.5

#: Subtracted from a node's own value to score its unvisited children. Without it, at 48
#: simulations, the first child visited keeps its lead purely because everything else is
#: scored at the optimistic default.
FPU_REDUCTION = 0.25

#: Root exploration noise, as AlphaZero. ``alpha`` is scaled to the branching factor:
#: ~10/moves is the usual rule, and Catan's ~20 legal moves gives 0.5.
DIRICHLET_ALPHA = 0.5
DIRICHLET_WEIGHT = 0.25


class Node:
    """One position in the tree.

    ``child_n``/``child_w`` are indexed by *position in* ``actions``, not by action index —
    a decision offering 12 moves allocates 12 floats, not 325.
    """

    __slots__ = ("state", "player", "kind", "actions", "children", "prior",
                 "child_n", "child_w", "child_q", "visits", "value", "winner")

    def __init__(self, state, kind, player, winner=None):
        self.state = state
        self.kind = kind
        self.player = player
        self.winner = winner
        #: Legal action indices as a numpy array, ascending (DECISION only). An array rather
        #: than a list so the prior can be gathered with one fancy index — a Python loop over
        #: `probabilities[i]` measured at 6.8 us against 0.2 us for the same gather.
        self.actions = None
        self.children = {}           # key -> Node; key is a slot for DECISION, a roll for CHANCE
        self.prior = None            # set when expanded
        self.child_n = None
        self.child_w = None
        #: ``child_w / child_n``, maintained on every backup instead of divided on every
        #: selection. Selection happens several times per simulation and backup once per
        #: edge, and the division carried a `np.errstate` block that alone cost more than the
        #: rest of the rule: 7.5 us a call against 2.7.
        self.child_q = None
        self.visits = 0
        self.value = 0.0

    @property
    def expanded(self):
        return self.prior is not None


def _settle(state, max_turns):
    """Apply forced moves until a real decision, a roll, or the end of the game.

    Returns ``(kind, legal_indices)``. Mutates ``state`` — callers pass a state they own.
    """
    while True:
        if state.phase is Phase.GAME_OVER:
            return TERMINAL, None
        if state.turn_number >= max_turns:
            return TERMINAL, None
        legal = action_space.legal_indices(state)
        if not legal:
            # No action available. The only way that is not a bug is a roll waiting to
            # happen; anything else means the engine offered nothing, which is a stuck game.
            if state.phase is Phase.ROLL:
                return CHANCE, None
            return TERMINAL, None
        if len(legal) == 1:
            rules.apply(state, action_space.decode(legal[0]))
            continue
        return DECISION, legal


def _node_for(state, max_turns, settle=True):
    """Wrap a state — already owned by the caller — as a node.

    ``settle=False`` classifies the position without playing forced moves through it. The
    root is built that way, because a search is asked "what should I do *here*"; collapsing
    the root would answer a question about a later position and the caller would apply the
    returned index to the wrong state.
    """
    if settle:
        kind, legal = _settle(state, max_turns)
    else:
        kind, legal = _classify(state, max_turns)
    if kind is TERMINAL:
        winner = state.winner if state.phase is Phase.GAME_OVER else None
        # At GAME_OVER `current_player` *becomes the winner*, so it is not a seat that acts.
        # The node's player is only used to orient a value, and a terminal value is oriented
        # by `winner`, so seat 1 is a placeholder, not a claim.
        return Node(state, TERMINAL, 1, winner=winner)
    node = Node(state, kind, state.current_player)
    if kind is DECISION:
        node.actions = np.asarray(legal, dtype=np.int64)
    return node


def _classify(state, max_turns):
    """What kind of position this is, without changing it."""
    if state.phase is Phase.GAME_OVER or state.turn_number >= max_turns:
        return TERMINAL, None
    legal = action_space.legal_indices(state)
    if legal:
        return DECISION, legal
    return (CHANCE, None) if state.phase is Phase.ROLL else (TERMINAL, None)


class Search:
    """One PUCT search, driven from outside so its evaluations can be batched.

        search = Search(world, budget=48, rng=rng)
        while (pending := search.request()) is not None:
            probabilities, value = evaluate(*pending)
            search.deliver(probabilities, value)
        counts = search.visit_counts()

    Args:
        state: the position to search. **Must already be determinized** — the search does not
            do it, because the caller usually wants to determinize once and reuse the world.
            Owned by the search from here on.
        budget: simulations. Every one costs at most one network evaluation.
        rng: :class:`numpy.random.Generator`, for the root noise and nothing else. Dice come
            from the state's own generator.
        c_puct, fpu, noise: see the module constants. ``noise=0`` disables root exploration,
            which is what evaluation and play want.
        max_turns: a game this long is scored as a draw rather than searched further.
    """

    def __init__(self, state, budget=48, rng=None, c_puct=C_PUCT, fpu=FPU_REDUCTION,
                 noise=DIRICHLET_WEIGHT, alpha=DIRICHLET_ALPHA, max_turns=400):
        if state.num_players != 2:
            raise ValueError("this search propagates a zero-sum scalar in seat 1's frame, "
                             "which is only meaningful for two players")
        self.rng = np.random.default_rng() if rng is None else rng
        self.c_puct = c_puct
        self.fpu = fpu
        self.noise = noise
        self.alpha = alpha
        self.max_turns = max_turns
        self.budget = budget

        self.root = _node_for(state, max_turns, settle=False)
        self.simulations = 0
        self._path = None            # [(node, key)] awaiting a value
        self._pending = None         # the node awaiting a policy

    # ------------------------------------------------------------------ #
    # The player-facing result                                            #
    # ------------------------------------------------------------------ #

    @property
    def player(self):
        return self.root.player

    @property
    def searchable(self):
        """Whether searching this root can change anything.

        A finished game and a chance node have nothing to decide. A root with exactly one
        legal move has a decision but no choice: spending the budget on it would buy a
        one-hot policy target that teaches nothing, so the driver plays it directly. Roughly
        a third of Catan's decision points are like this.
        """
        return self.root.kind is DECISION and len(self.root.actions) > 1

    @property
    def forced(self):
        """The only legal action, when there is exactly one. ``None`` otherwise."""
        if self.root.kind is DECISION and len(self.root.actions) == 1:
            return int(self.root.actions[0])
        return None

    def visit_counts(self):
        """``(actions, counts)`` at the root — the improved policy AlphaZero learns from."""
        if self.root.kind is not DECISION or not self.root.expanded:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        return (np.asarray(self.root.actions, dtype=np.int64),
                np.asarray(self.root.child_n, dtype=np.float64))

    def root_value(self):
        """The search's opinion of the root, in the root player's frame."""
        if self.root.visits == 0:
            return self.root.value
        return float(np.sum(self.root.child_w) / self.root.visits)

    def best_action(self, temperature=0.0):
        """Sample a move from the visit counts. ``temperature=0`` takes the most visited."""
        if self.root.kind is not DECISION:
            raise RuntimeError("nothing to choose: the root is not a decision")
        actions = np.asarray(self.root.actions, dtype=np.int64)
        if not self.root.expanded:
            return int(actions[0])                  # never searched: only sane for a forced move
        counts = np.asarray(self.root.child_n, dtype=np.float64)
        if counts.sum() == 0:
            return int(actions[int(np.argmax(self.root.prior))])
        if temperature <= 1e-3:
            return int(actions[int(np.argmax(counts))])
        weights = counts ** (1.0 / temperature)
        weights /= weights.sum()
        return int(actions[int(self.rng.choice(len(actions), p=weights))])

    # ------------------------------------------------------------------ #
    # The evaluation protocol                                             #
    # ------------------------------------------------------------------ #

    def request(self):
        """Descend until a position needs the network, or the budget is spent.

        Returns ``(observation, mask)``, or ``None`` when this search is finished. Terminal
        lines are resolved and backed up inside the loop and consume a simulation each, so a
        search whose subtree is entirely decided finishes without a single evaluation.
        """
        if self._pending is not None:
            raise RuntimeError("a previous request has not been delivered")
        if not self.searchable:
            return None
        while self.simulations < self.budget:
            node, path = self._descend()
            if node is None:
                continue                     # terminal line: already backed up
            self._pending, self._path = node, path
            # The mask is built from the legality this node already computed, not by asking
            # the rules again: `legal_mask` and `legal_indices` each cost about 42 us on a
            # real position, and calling both at every leaf spent a ninth of the search
            # deriving the same fact twice.
            mask = np.zeros(action_space.NUM_ACTIONS, dtype=bool)
            mask[node.actions] = True
            observation = np.fromiter(encoder.encode(node.state, node.player),
                                      dtype=np.float32, count=encoder.SIZE)
            return observation, mask
        return None

    def deliver(self, probabilities, value):
        """Expand the pending leaf with a prior and back its value up the path.

        Args:
            probabilities: over the whole action space, already masked and normalised.
            value: the leaf position in **its own player's** frame, in ``[-1, 1]``.
        """
        node, path = self._pending, self._path
        if node is None:
            raise RuntimeError("deliver() without a matching request()")
        self._pending = self._path = None

        prior = np.asarray(probabilities, dtype=np.float64)[node.actions]
        total = prior.sum()
        # A leaf whose legal moves all got zero probability would make the PUCT term vanish
        # and the search would degenerate to visiting slot 0. Uniform is the honest fallback.
        prior = prior / total if total > 1e-12 else np.full(len(node.actions),
                                                            1.0 / len(node.actions))
        if node is self.root and self.noise > 0:
            draw = self.rng.dirichlet([self.alpha] * len(prior))
            prior = (1 - self.noise) * prior + self.noise * draw

        node.prior = prior
        node.child_n = np.zeros(len(prior), dtype=np.float64)
        node.child_w = np.zeros(len(prior), dtype=np.float64)
        node.value = float(value)
        # Unvisited children are scored from this node's own value, reduced. A default of 0
        # would make every unvisited child look better than a losing position and worse than
        # a winning one, so the search would explore hardest exactly where it is behind.
        node.child_q = np.full(len(prior), node.value - self.fpu, dtype=np.float64)

        self._backup(path, self._orient(float(value), node.player))
        self.simulations += 1

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _orient(value, player):
        """Convert between ``player``'s frame and seat 1's.

        Zero-sum and two-player, so the conversion is its own inverse and one function does
        both directions. Written once so the two call sites cannot drift apart — a sign error
        in one of them is the classic silent failure here: the search would steer toward
        positions good for the opponent and would still look like it was working.
        """
        return value if player == 1 else -value

    def _descend(self):
        """One simulation's walk from the root.

        Returns ``(leaf, path)`` when the network is needed, or ``(None, None)`` when the
        walk ended in a decided position — in which case the value has already been backed
        up and the simulation counted.
        """
        node = self.root
        path = []
        while True:
            if node.kind is TERMINAL:
                self._backup(path, self._terminal_value(node))
                self.simulations += 1
                return None, None
            if node.kind is CHANCE:
                # Chance nodes take no part in `path`: they hold no statistics to update, and
                # the value flows through them unchanged because the frame is absolute.
                node = self._sample_roll(node)
                continue
            if not node.expanded:
                return node, path
            slot = self._select(node)
            path.append((node, slot))
            child = node.children.get(slot)
            if child is None:
                world = node.state.clone(rng=node.state.rng)
                rules.apply(world, action_space.decode(node.actions[slot]))
                child = _node_for(world, self.max_turns)
                node.children[slot] = child
            node = child

    def _terminal_value(self, node):
        """A finished game, in seat 1's frame. A truncated game is a draw, worth 0.

        Truncation must not be scored as a loss for either side: a policy that cannot finish
        has failed, but attributing that failure to whoever happened to be on move would teach
        the other seat that stalling wins.
        """
        if node.winner is None:
            return 0.0
        return 1.0 if node.winner == 1 else -1.0

    def _sample_roll(self, node):
        """Draw a roll and step into the child for that total, creating it if new.

        Keyed by the total rather than the pair, because 5+2 and 3+4 lead to the same game.
        Re-sampling on revisit is what makes a child's share of the visits match how often
        that total actually comes up.
        """
        world = node.state.clone(rng=node.state.rng)
        roll = rules.roll_dice(world)
        child = node.children.get(roll)
        if child is None:
            child = _node_for(world, self.max_turns)
            node.children[roll] = child
        return child

    def _select(self, node):
        """PUCT. Returns the slot to descend into."""
        weight = self.c_puct * math.sqrt(node.visits if node.visits else 1)
        return int(np.argmax(
            node.child_q + weight * node.prior / (1.0 + node.child_n)
        ))

    def _backup(self, path, value_seat_one):
        """Credit every edge on the path, each in its own node's frame."""
        for node, slot in path:
            node.visits += 1
            node.child_n[slot] += 1.0
            node.child_w[slot] += self._orient(value_seat_one, node.player)
            node.child_q[slot] = node.child_w[slot] / node.child_n[slot]
