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

#: Root actions sampled without replacement by the Gumbel root. 16 against a measured mean of
#: 9.6 legal moves means the sample is usually the whole set, and the halving schedule rather
#: than the sampling is what does the work.
GUMBEL_ACTIONS = 16

#: The monotone transform applied to Q before it is added to the logits, from Danihelka et al.
#: ``sigma(q) = (c_visit + max_visits) * c_scale * q``. Scaling by the largest visit count is
#: what makes the transform grow with the search: early on the prior dominates, and by the end
#: the Q values do.
C_VISIT = 50.0
C_SCALE = 1.0


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
        gumbel: use the Gumbel root of Danihelka et al. instead of PUCT-plus-Dirichlet at
            the root. Off by default so every existing caller is unchanged.

            **Why it exists here.** Measured on this repository: at 96 simulations the
            visit-count target agrees with a clean 400-simulation search 80.5% of the time
            and the network's own prior already agrees 76.2% — +4.3 points, down from the
            +20 that justified the setting in record 0023, and not significant once the
            five settings compared are accounted for. Positions offer 9.6 legal moves on
            average, so 96 PUCT simulations spend most of their budget confirming the
            prior, and the visit counts come back as a copy of it. Sequential Halving
            spends the same budget *discriminating* between a sampled set instead, and the
            completed-Q target uses the value estimate for every action rather than only
            the visited ones, so the target can differ from the prior even where the visits
            do not. The construction is a policy improvement at any budget, which plain
            visit counts are not.
        gumbel_actions, c_visit, c_scale: see the module constants.
    """

    def __init__(self, state, budget=48, rng=None, c_puct=C_PUCT, fpu=FPU_REDUCTION,
                 noise=DIRICHLET_WEIGHT, alpha=DIRICHLET_ALPHA, max_turns=400,
                 gumbel=False, gumbel_actions=GUMBEL_ACTIONS, c_visit=C_VISIT,
                 c_scale=C_SCALE):
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
        self.gumbel = bool(gumbel)
        self.gumbel_actions = int(gumbel_actions)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)

        self.root = _node_for(state, max_turns, settle=False)
        self.simulations = 0
        self._path = None            # [(node, key)] awaiting a value
        self._pending = None         # the node awaiting a policy
        #: Scratch for leaf encoding. Per-search, not shared: several searches are in flight
        #: at once inside one :class:`~training.alphazero.self_play.Generator`.
        self._buffer = encoder.observation_buffer()

        # --- Gumbel root state, all set when the root is expanded --------------------- #
        self._gumbel = None           # one Gumbel(0,1) draw per root action
        self._logits = None           # log of the root prior, before any noise
        self._candidates = None       # slots still in contention, as a numpy array
        self._phase_counts = None     # visits given in the current phase, per slot
        self._phase_quota = 0
        self._phases = 1

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
        if self.gumbel:
            # The Gumbel variables *are* the exploration, so this is stochastic across
            # searches while being deterministic within one. ``temperature`` is therefore
            # ignored rather than silently combined with it, which would be exploration
            # applied twice and would undo the improvement guarantee.
            return int(actions[int(self._candidates[
                np.argmax(self._root_scores()[self._candidates])])])
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
            # Encoded straight into float32 rather than into a list of Python floats that is
            # then unboxed element by element. `np.fromiter` over 2,503 floats measured 42 us
            # against 95 us for the encode itself, once per simulation; `np.frombuffer` over
            # the same numbers already stored as float32 is 0.5 us.
            #
            # The `.copy()` is not optional. `Generator.run` holds one pending observation
            # per game in flight and stacks them at the end of the round, so a view onto a
            # buffer this search will overwrite is a row that silently becomes some later
            # position. It costs 0.5 us and removes the hazard rather than documenting it.
            encoder.encode_into(node.state, node.player, self._buffer)
            observation = np.frombuffer(self._buffer, dtype=np.float32).copy()
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
        if node is self.root and self.gumbel:
            # Gumbel *replaces* Dirichlet: the noise enters through the argmax rather than
            # through the prior, which is what makes the result a policy improvement rather
            # than a perturbed prior. Adding both would be exploration twice, and record
            # 0023 already measured Dirichlet at 0.25 flipping 24% of the top moves.
            self._setup_gumbel(prior)
        elif node is self.root and self.noise > 0:
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
    # The Gumbel root                                                     #
    # ------------------------------------------------------------------ #

    def _setup_gumbel(self, prior):
        """Draw the Gumbel variables and open the first Sequential Halving phase.

        ``logits`` is the log of the *normalised* prior. Any constant shift is common to
        every action, and every use below is either a softmax or an argmax over actions, so
        the missing normalisation constant cancels exactly.
        """
        n = len(prior)
        self._logits = np.log(np.clip(prior, 1e-12, None))
        self._gumbel = self.rng.gumbel(size=n)
        m = min(self.gumbel_actions, n)
        # Gumbel-Top-k: the top m of (logits + gumbel) is a sample of m actions drawn
        # without replacement from the prior. This is the sampling step, done once.
        self._candidates = np.sort(np.argsort(self._logits + self._gumbel)[::-1][:m])
        self._phase_counts = np.zeros(n, dtype=np.int64)
        self._phases = max(1, int(math.ceil(math.log2(m))) if m > 1 else 1)
        self._open_phase()

    def _open_phase(self):
        """Visits each surviving candidate gets before the set is halved again."""
        self._phase_counts[:] = 0
        share = self.budget // (self._phases * max(1, len(self._candidates)))
        self._phase_quota = max(1, int(share))

    def _sigma(self, q):
        """The monotone transform of Q that competes with the logits."""
        largest = float(self.root.child_n.max()) if self.root.child_n is not None else 0.0
        return (self.c_visit + largest) * self.c_scale * q

    def _root_scores(self):
        """``gumbel + logits + sigma(q)`` per root slot, for halving and for the final pick.

        Unvisited slots score on ``gumbel + logits`` alone. Inside a phase every candidate
        has at least one visit by construction, so this only matters for slots that were
        never sampled, and those must never win.
        """
        node = self.root
        visited = node.child_n > 0
        return np.where(visited, self._gumbel + self._logits + self._sigma(node.child_q),
                        self._gumbel + self._logits)

    def _select_root_gumbel(self):
        """The next root slot to visit, under Sequential Halving."""
        while True:
            counts = self._phase_counts[self._candidates]
            slot = int(np.argmin(counts))
            if counts[slot] < self._phase_quota:
                chosen = int(self._candidates[slot])
                self._phase_counts[chosen] += 1
                return chosen
            if len(self._candidates) <= 1:
                # Budget left over and one candidate standing: keep spending on it. The
                # extra visits sharpen its Q, which the policy target reads.
                self._phase_quota += 1
                continue
            keep = max(1, len(self._candidates) // 2)
            scores = self._root_scores()[self._candidates]
            survivors = self._candidates[np.argsort(scores)[::-1][:keep]]
            self._candidates = np.sort(survivors)
            self._open_phase()

    def policy_target(self):
        """``(actions, probabilities)`` — the improved policy to train on.

        Plain AlphaZero learns from normalised visit counts. Under the Gumbel root that
        would throw away most of what the search found: with 96 simulations over a handful
        of candidates, the visit counts are nearly uniform over the survivors by
        construction, because Sequential Halving *deliberately* spends its budget evenly.

        So the target is ``softmax(logits + sigma(completedQ))`` over **all** legal actions,
        with ``completedQ(a) = q(a)`` where the action was visited and ``v_mix`` where it
        was not. ``v_mix`` is the search's own estimate of the root, blended with the value
        of the actions it did visit, weighted by their prior mass — so an unvisited action
        is scored as "about what this position is worth" rather than as unknown.
        """
        node = self.root
        if node.kind is not DECISION or not node.expanded:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        actions = np.asarray(node.actions, dtype=np.int64)
        if not self.gumbel:
            counts = np.asarray(node.child_n, dtype=np.float64)
            total = counts.sum()
            if total <= 0:
                return actions, np.asarray(node.prior, dtype=np.float64)
            return actions, counts / total

        visited = node.child_n > 0
        if not visited.any():
            return actions, np.asarray(node.prior, dtype=np.float64)
        total_n = float(node.child_n.sum())
        weight = float(node.prior[visited].sum())
        weighted_q = float(np.sum(node.prior[visited] * node.child_q[visited]))
        v_mix = (node.value + (total_n / weight) * weighted_q) / (1.0 + total_n)

        completed = np.where(visited, node.child_q, v_mix)
        scores = self._logits + self._sigma(completed)
        scores -= scores.max()
        weights = np.exp(scores)
        return actions, weights / weights.sum()

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
            slot = (self._select_root_gumbel()
                    if self.gumbel and node is self.root else self._select(node))
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
        """Step into the child for the roll this node produces, creating it if new.

        Keyed by the total rather than the pair, because 5+2 and 3+4 lead to the same game.

        **Under plain dice, revisiting re-samples**, which is what makes a child's share of
        the visits match how often that total actually comes up.

        **Under the Balanced Dice deck it does not, and cannot.** The deck is consumed rather
        than resampled: :func:`catan.dice.draw_balanced` pops the *last* card, and
        :meth:`~catan.state.GameState.clone` copies the deck verbatim, so every clone of this
        node draws the same card. Measured over 9,002 real chance nodes under ``RANKED_1V1``,
        24 resamples of each produced **one** distinct total every time; the same measurement
        under ``BASE_GAME`` gives 5-11. So the clone and the roll on a revisit were computing
        a number already visible in ``dice_deck[-1]`` — 77% of chance-node visits, and 7.3%
        of self-play wall clock, spent re-deriving it.

        The fast path therefore only skips work whose answer is pinned. The miss path below
        is unchanged, and ``dice_deck is None`` under plain dice makes the guard fall straight
        through to it.

        ⚠️ This changes which game a seed produces, and it is worth knowing why: a discarded
        clone's deck could fall to ``RESHUFFLE_AT`` and call ``new_deck(state.rng)`` on the
        *shared* generator, so those throw-away clones were advancing the real game's random
        stream. Not rolling them leaves the stream where it was. The games are drawn from the
        same distribution; they are not the same games.
        """
        deck = node.state.dice_deck
        if deck:
            first, second = deck[-1]
            child = node.children.get(first + second)
            if child is not None:
                return child

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
