"""The training pipeline.

PPO fails silently. A run with a sign error in the advantage, or a mask that differs by one
bit between rollout and update, does not crash — it trains for hours and produces a policy
that is merely bad, which is indistinguishable from a policy that needs more hours. So the
things worth testing here are the ones no amount of watching a loss curve would reveal.

Skipped entirely if PyTorch is not installed: the engine has no dependencies and must stay
testable without one.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training needs PyTorch")

from catan import action_space, encoder                                  # noqa: E402
from catan.agents import HeuristicAgent, RandomAgent                     # noqa: E402
from catan.env import CatanEnv                                           # noqa: E402
from catan.rulesets import RANKED_1V1                                    # noqa: E402
from training.agent import PolicyAgent                                   # noqa: E402
from training.evaluate import confidence_interval                        # noqa: E402
from training.net import MASK_FILL, PolicyValueNet                       # noqa: E402
from training.pool import OpponentPool                                   # noqa: E402
from training.ppo import PPO, explained_variance                         # noqa: E402
from training.rollout import (                                           # noqa: E402
    SelfPlayCollector,
    compute_gae,
    outcome,
    potential,
)


@pytest.fixture
def net():
    torch.manual_seed(0)
    return PolicyValueNet(encoder.SIZE, action_space.NUM_ACTIONS, hidden=(64, 64))


@pytest.fixture
def rollout(net):
    collector = SelfPlayCollector(net, num_envs=8, max_turns=120, seed=3, shaping=0.0)
    return collector.collect(400)


# =========================================================================== #
# MASKING — where a one-bit disagreement destroys a run                       #
# =========================================================================== #

def test_illegal_actions_get_zero_probability(net):
    obs = torch.zeros(1, encoder.SIZE)
    mask = torch.zeros(1, action_space.NUM_ACTIONS, dtype=bool)
    mask[0, [3, 17, 200]] = True

    with torch.no_grad():
        logits, _ = net(obs)
    masked = net._apply_mask(logits, mask)
    probabilities = torch.softmax(masked, dim=-1)[0]

    assert probabilities[~mask[0]].sum() == 0.0, "an illegal action had probability"
    assert probabilities[mask[0]].sum() == pytest.approx(1.0, abs=1e-5)


def test_the_mask_fill_is_finite():
    """``-inf`` would give ``0 * inf = nan`` in the entropy of a one-action state."""
    assert np.isfinite(MASK_FILL)

    net = PolicyValueNet(encoder.SIZE, action_space.NUM_ACTIONS, hidden=(16,))
    mask = torch.zeros(1, action_space.NUM_ACTIONS, dtype=bool)
    mask[0, 5] = True
    logp, entropy, _ = net.evaluate(torch.zeros(1, encoder.SIZE), mask, torch.tensor([5]))
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    assert entropy.item() == pytest.approx(0.0, abs=1e-4)


def test_act_never_returns_an_illegal_action(net):
    torch.manual_seed(1)
    mask = torch.zeros(64, action_space.NUM_ACTIONS, dtype=bool)
    for row in range(64):
        mask[row, np.random.default_rng(row).choice(324, 4, replace=False)] = True
    action, _, _ = net.act(torch.randn(64, encoder.SIZE), mask)
    assert mask[torch.arange(64), action].all()


def test_stored_actions_rescore_to_the_stored_log_probabilities(net, rollout):
    """The PPO ratio is ``exp(new - old)``. If re-scoring an unchanged network does not
    reproduce ``old``, the ratio is wrong on the very first minibatch and everything after
    it is noise."""
    logp, _, value = net.evaluate(rollout.obs, rollout.mask, rollout.action)
    assert torch.allclose(logp, rollout.logp, atol=1e-4), "rollout and update disagree"
    assert torch.allclose(value, rollout.value, atol=1e-4)


def test_every_stored_action_was_legal(rollout):
    assert rollout.mask[torch.arange(len(rollout)), rollout.action].all()


def test_forced_decisions_are_not_stored(rollout):
    """They carry no gradient — the ratio is 1 by construction — so they would only
    dilute the batch."""
    assert (rollout.mask.sum(dim=1) > 1).all()


# =========================================================================== #
# CREDIT ASSIGNMENT — the loser has to be told it lost                        #
# =========================================================================== #

def test_the_winner_gains_and_the_loser_loses():
    info = {"winner": 1, "public_scores": {1: 15, 2: 8}}
    assert outcome(info, 1, 15) == 1.0
    assert outcome(info, 2, 15) == -1.0


def test_an_unfinished_game_is_adjudicated_on_points():
    ahead = {"winner": None, "public_scores": {1: 9, 2: 4}}
    behind = {"winner": None, "public_scores": {1: 4, 2: 9}}
    level = {"winner": None, "public_scores": {1: 6, 2: 6}}

    assert outcome(ahead, 1, 15) > 0
    assert outcome(behind, 1, 15) < 0
    assert outcome(level, 1, 15) == 0.0
    # never worth as much as actually winning, or stalling would be a strategy
    assert abs(outcome(ahead, 1, 15)) < 1.0


def test_the_adjudication_is_zero_sum():
    info = {"winner": None, "public_scores": {1: 11, 2: 3}}
    assert outcome(info, 1, 15) == pytest.approx(-outcome(info, 2, 15))


def test_potential_reads_only_public_points():
    """A shaping term that saw victory-point development cards would leak them."""
    info = {"winner": None, "public_scores": {1: 7, 2: 2}}
    assert potential(info, 1, 15) == pytest.approx(5 / 15)
    assert potential(info, 2, 15) == pytest.approx(-5 / 15)


def test_gae_spreads_a_terminal_reward_backwards():
    rewards = np.array([0, 0, 0, 1.0], dtype=np.float32)
    values = np.zeros(4, dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)

    assert advantages[-1] == pytest.approx(1.0)
    for t in range(3):
        assert advantages[t] == pytest.approx(0.95 ** (3 - t))
    assert np.all(np.diff(advantages) > 0), "credit must grow toward the decisive move"
    assert np.allclose(returns, advantages)          # values are zero


def test_gae_with_a_perfect_critic_gives_no_advantage():
    values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rewards = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # a critic already predicting 1 everywhere, and the episode paying 0 at the end,
    # should produce the shortfall, not zero — the last state was over-valued
    advantages, _ = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    assert advantages[-1] == pytest.approx(-1.0)


def test_a_losing_seat_gets_negative_returns(net):
    """The property the whole per-seat machinery exists for: the environment never hands
    back the loser's reward, so if this fails the agent is training on winners only."""
    collector = SelfPlayCollector(net, num_envs=12, max_turns=400, seed=11, shaping=0.0)
    rollout = collector.collect(3_000)
    returns = rollout.returns.numpy()
    assert returns.min() < -0.5, "nobody was ever told they lost"
    assert returns.max() > 0.5, "nobody was ever told they won"
    assert abs(float(returns.mean())) < 0.35, (
        f"self-play should be roughly zero-sum, got mean {returns.mean():.3f}"
    )


def test_both_seats_are_learned_from_in_self_play(net):
    collector = SelfPlayCollector(net, num_envs=6, max_turns=150, seed=5)
    before = collector.collect(600)
    assert before.stats["games"] > 0
    # both seats banked => roughly twice the transitions per game of a one-seat collector
    per_game = len(before) / before.stats["games"]
    assert per_game > 20, per_game


# =========================================================================== #
# THE UPDATE                                                                  #
# =========================================================================== #

def test_an_update_changes_the_policy_and_reports_its_diagnostics(net, rollout):
    before = [p.detach().clone() for p in net.parameters()]
    ppo = PPO(net, minibatch=64, epochs=2)
    diagnostics = ppo.update(rollout)

    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    for key in ("policy_loss", "value_loss", "entropy", "kl",
                "clip_fraction", "explained_variance"):
        assert key in diagnostics and np.isfinite(diagnostics[key]), key
    assert diagnostics["kl"] >= -1e-6, "the KL estimator went negative"
    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0


def test_the_critic_learns_a_constant_target(net, rollout):
    """The cheapest possible check that gradients flow the right way: make every return
    the same number and see whether the value head finds it."""
    rollout.returns = torch.full_like(rollout.returns, 0.7)
    rollout.advantage = torch.zeros_like(rollout.advantage)
    ppo = PPO(net, minibatch=128, epochs=8, lr=3e-3, value_clip=None, entropy_coef=0.0)
    first = ppo.update(rollout)["value_loss"]
    for _ in range(6):
        last = ppo.update(rollout)["value_loss"]
    assert last < first, f"value loss did not fall: {first:.4f} -> {last:.4f}"


def test_explained_variance_means_what_it_says():
    actual = np.array([1.0, -1.0, 1.0, -1.0])
    assert explained_variance(actual, actual) == pytest.approx(1.0)
    assert explained_variance(np.zeros(4), actual) == pytest.approx(0.0)
    assert explained_variance(-actual, actual) < 0
    assert explained_variance(np.ones(4), np.ones(4)) == 0.0     # no variance to explain


# =========================================================================== #
# THE AGENT AND THE POOL                                                      #
# =========================================================================== #

def test_a_policy_agent_is_a_callable_like_every_other_agent(net):
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=1)
    agent = PolicyAgent(net)
    for _ in range(200):
        if info["done"]:
            break
        action = agent(observation, info)
        assert action in info["legal"]
        observation, _, _, _, info = env.step(action)


def test_a_policy_agent_survives_a_round_trip(net, tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, path)
    loaded = PolicyAgent.load(path)

    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=2)
    assert loaded(observation, info) == PolicyAgent(net)(observation, info)


def test_temperature_zero_is_deterministic(net):
    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=4)
    agent = PolicyAgent(net, temperature=0.0)
    assert len({agent(observation, info) for _ in range(10)}) == 1


def test_the_pool_offers_self_frozen_and_the_heuristic(net):
    pool = OpponentPool(self_play=0.4, heuristic=0.3, seed=1)
    for iteration in range(3):
        pool.add(net, iteration)

    labels = {pool.sample(net, 10)[1].split("@")[0] for _ in range(200)}
    assert labels == {"self", "heuristic", "frozen"}, labels


def test_the_pool_stays_within_capacity(net):
    pool = OpponentPool(capacity=4, seed=0)
    for iteration in range(20):
        pool.add(net, iteration)
    assert len(pool) == 4
    kept = [i for i, _ in pool.frozen]
    assert kept[0] == 0, "the oldest checkpoint is the widest test; keep it"
    assert kept[-1] == 19, "the newest is the hardest; keep it"


def test_a_frozen_opponent_is_a_snapshot_not_a_reference(net):
    """If the pool held a reference, every 'past self' would silently be the current one
    and the whole anti-forgetting argument would be void."""
    pool = OpponentPool(seed=0)
    pool.add(net, 0)
    frozen, _ = pool.frozen[0][1], None
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(1.0)
    assert not torch.allclose(
        frozen["policy_head.weight"], net.policy_head.weight
    ), "the pool holds a live reference to the training network"


# =========================================================================== #
# EVALUATION                                                                  #
# =========================================================================== #

def test_the_confidence_interval_narrows_with_more_games():
    narrow = confidence_interval(50, 100)
    narrower = confidence_interval(500, 1000)
    assert (narrower[1] - narrower[0]) < (narrow[1] - narrow[0])
    assert narrow[0] < 0.5 < narrow[1]


def test_the_interval_stays_inside_zero_and_one():
    """The reason for Wilson rather than the normal approximation, which does not."""
    low, high = confidence_interval(100, 100)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert low > 0.9

    low, high = confidence_interval(0, 50)
    assert low == 0.0 and 0.0 < high < 0.2


# =========================================================================== #
# ENGINE INVARIANTS THE TRAINER DEPENDS ON                                    #
# =========================================================================== #
#
# Three properties of CatanEnv that the rollout quietly relies on. None is
# documented as a guarantee, all three are load-bearing, and each fails
# silently — a run would train happily and learn nothing. They were surfaced by
# measuring the engine rather than reading it, so they are pinned here.

def _decided_games(count=12, max_turns=600):
    for seed in range(count):
        env = CatanEnv(num_players=2, max_turns=max_turns)
        obs, info = env.reset(seed=seed)
        agents = {1: HeuristicAgent(seed), 2: HeuristicAgent(seed + 99)}
        while not info["done"]:
            actor = info["player"]
            obs, reward, terminated, _, info = env.step(agents[actor](obs, info))
            if terminated:
                yield env, obs, reward, actor, info
                break


def test_the_winner_is_always_the_player_who_just_acted():
    """So ``step()``'s reward is *always* +1 and ``LOSS_REWARD`` is unreachable on the
    normal path. A learner that consumed the returned reward would see a constant +1, the
    critic would converge to V = 1, every advantage would collapse to zero, and nothing
    would crash. The rollout reads ``info["winner"]`` and writes both seats instead."""
    seen = list(_decided_games())
    assert seen, "no game finished"
    assert {r for _, _, r, _, _ in seen} == {1.0}
    for _, _, _, actor, info in seen:
        assert info["winner"] == actor


def test_the_terminal_observation_belongs_to_the_winner():
    """``state.current_player`` becomes the winner once the phase is GAME_OVER, so the
    observation ``step`` hands back at termination is the *winner's* view. Closing the
    loser's trajectory by bootstrapping from it would be an exact sign flip on half the
    data. The rollout stores the pre-step observation and bootstraps nothing at a
    terminal."""
    for env, obs, _, _, info in _decided_games():
        loser = 3 - info["winner"]
        assert obs == env.observe(info["winner"])
        assert obs != env.observe(loser)


def test_the_terminal_mask_is_empty():
    """Nothing is legal once the game is over, so a masked softmax over it is a softmax
    over nothing. The collector resets the environment the moment a game ends, so a
    terminal ``info`` is never handed to the network."""
    for _, _, _, _, info in _decided_games():
        assert sum(info["mask"]) == 0


def test_the_collector_never_queries_the_network_on_a_dead_game(net):
    """The consequence of the three above, checked end to end."""
    collector = SelfPlayCollector(net, num_envs=10, max_turns=200, seed=7)
    for _ in range(4):
        collector.collect(300)
        for info in collector.info:
            assert not info["done"], "a finished game was left in the pool"
            assert sum(info["mask"]) > 0, "a dead position would reach the policy head"
