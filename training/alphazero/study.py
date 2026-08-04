"""What opening positions actually win, measured rather than assumed.

    python -m training.alphazero.study --games 300

Plays an agent against a fixed opponent and records, for every game, what its **opening
placements were** and whether it went on to win. That turns "the AI places badly" from an
impression into a table: which pip totals win, which resource mixes win, whether being on a
harbour matters, and whether the ore-wheat-sheep opening or the expansion opening does better
*on this ruleset*.

**It runs in a separate process from training and touches nothing training owns.** It reads
``models/`` like any other player and writes a JSON file. Start it while a run is going and
the run does not notice, beyond the cores it takes.

**What is recorded per game**

``pips``            expected cards per roll across both opening settlements
``best_pips``       the most that was available when each was placed — the gap is the
                    question "did it take the best spot", and taking the highest-pip spot
                    every time is not the same as playing well
``resources``       which resources the opening produces, and how much of each
``diversity``       how many distinct resources, which is what "an actionable hand" means
                    before any cards have been drawn
``harbour``         whether either settlement sits on one
``ore_wheat_sheep`` share of the opening's production in the development-card resources
``wood_brick``      share in the expansion resources — the other classic opening
``won``             the outcome

The two strategy shares are deliberately *descriptive*, not prescriptive: nothing in the
engine or the agent knows they exist. They are computed here so a person can ask "does the
ore-wheat-sheep opening win more often on a 15-point 1v1 board" and get a number.
"""

import argparse
import collections
import json
import pathlib

from catan import rules
from catan.actions import ActionType
from catan.board import GENERIC_HARBOUR
from catan.env import CatanEnv
from catan.resources import NUM_RESOURCES, Resource
from catan.rulesets import RANKED_1V1
from catan.topology import NUM_VERTICES, VERTEX_TILES

#: Where a study is written by default. Read by the dashboard.
STUDY = pathlib.Path("checkpoints/opening_study.json")

#: The development-card strategy's resources, and the expansion strategy's.
ORE_WHEAT_SHEEP = (Resource.ORE, Resource.WHEAT, Resource.SHEEP)
WOOD_BRICK = (Resource.WOOD, Resource.BRICK)


def odds(number):
    """Expected share of rolls that produce this number. 0 for a desert."""
    return 0.0 if number is None else (6 - abs(7 - number)) / 36.0


def vertex_production(board, vertex):
    """Expected cards per roll of each resource at ``vertex``."""
    per = [0.0] * NUM_RESOURCES
    for tile in VERTEX_TILES[vertex]:
        resource = board.resource_at(tile)
        if resource is not None:
            per[int(resource)] += odds(board.number_at(tile))
    return per


def best_available(state):
    """The most production on offer among the spots that are legal right now."""
    return max(
        (sum(vertex_production(state.board, v)) for v in range(1, NUM_VERTICES + 1)
         if rules.can_place_setup_settlement(state, v)),
        default=0.0,
    )


def play_and_record(agent, opponent, seed, ruleset=RANKED_1V1, max_turns=400):
    """One game. Returns the opening record and the outcome."""
    from catan import action_space

    env = CatanEnv(num_players=2, ruleset=ruleset, max_turns=max_turns)
    observation, info = env.reset(seed=seed)
    # Seats alternate with the seed so the study is not a study of going first.
    me = 1 if seed % 2 == 0 else 2
    seats = {me: agent, 3 - me: opponent}
    board = env.state.board

    production = [0.0] * NUM_RESOURCES
    gaps, spots, harbours = [], [], 0

    while not info["done"]:
        player = info["player"]
        action_index = seats[player](observation, info)
        action = action_space.decode(action_index)
        if (player == me and env.state.in_setup
                and action.type is ActionType.BUILD_SETTLEMENT):
            available = best_available(env.state)
            here = vertex_production(board, action.position)
            for resource in range(NUM_RESOURCES):
                production[resource] += here[resource]
            gaps.append(available - sum(here))
            spots.append(action.position)
            harbours += 1 if board.harbours_at(action.position) else 0
        observation, _, _, _, info = env.step(action_index)

    total = sum(production)
    owned = set()
    for vertex in range(1, NUM_VERTICES + 1):
        if env.state.vertex_owner[vertex] == me:
            for harbour in board.harbours_at(vertex):
                owned.add("generic" if harbour is GENERIC_HARBOUR
                          else Resource(harbour).name.lower())

    return {
        "seed": seed,
        "seat": me,
        "pips": round(total, 4),
        "gap_to_best": round(sum(gaps), 4),
        "spots": spots,
        "resources": {Resource(r).name.lower(): round(production[r], 4)
                      for r in range(NUM_RESOURCES)},
        "diversity": sum(1 for value in production if value > 0),
        "opening_harbours": harbours,
        "harbours_owned": sorted(owned),
        "ore_wheat_sheep": round(
            sum(production[int(r)] for r in ORE_WHEAT_SHEEP) / total, 4) if total else 0.0,
        "wood_brick": round(
            sum(production[int(r)] for r in WOOD_BRICK) / total, 4) if total else 0.0,
        "turns": env.state.turn_number,
        "won": info["winner"] == me,
        "decided": info["winner"] is not None,
    }


def run(agent, opponent, games=300, seed=90_000, log=print):
    """Play ``games`` and return the per-game records."""
    records = []
    for game in range(games):
        records.append(play_and_record(agent, opponent, seed + game))
        if log and (game + 1) % 25 == 0:
            won = sum(r["won"] for r in records)
            log(f"  {game + 1}/{games} games, {100 * won / len(records):.0f}% won")
    return records


# --------------------------------------------------------------------------- #
# Summaries a person can act on                                               #
# --------------------------------------------------------------------------- #

def quantile_edges(values, bands=4):
    """Band boundaries that split ``values`` into equal-sized groups.

    Fixed edges were tried first and were wrong in a way worth recording: an opening is
    *two* settlements, so its production is around 0.6, and edges chosen for one settlement
    put every game in the top band and reported a single row. Quantiles cannot make that
    mistake, and they also adapt when the agent's placements shift — which is the whole
    point of running this again after a training run.
    """
    ordered = sorted(values)
    if not ordered:
        return []
    edges = [ordered[0]]
    for band in range(1, bands):
        edges.append(ordered[min(len(ordered) - 1, band * len(ordered) // bands)])
    edges.append(ordered[-1] + 1e-9)
    # Collapse duplicates, which happen when a quantity is concentrated.
    return sorted(set(edges))


def bucket(records, key, edges, label):
    """Win rate by band of a continuous quantity."""
    out = []
    for low, high in zip(edges, edges[1:]):
        inside = [r for r in records if low <= r[key] < high and r["decided"]]
        if not inside:
            continue
        out.append({
            "band": f"{label} {low:.2f}-{high:.2f}",
            "low": low, "high": high,
            "games": len(inside),
            "win_rate": round(sum(r["won"] for r in inside) / len(inside), 4),
        })
    return out


def summarise(records):
    """Everything the dashboard draws, computed once here."""
    decided = [r for r in records if r["decided"]]
    if not decided:
        return {"games": len(records), "decided": 0}

    by_resource = {}
    for name in [r.name.lower() for r in Resource]:
        with_it = [r for r in decided if r["resources"][name] > 0]
        without = [r for r in decided if r["resources"][name] == 0]
        by_resource[name] = {
            "with": {"games": len(with_it),
                     "win_rate": round(sum(r["won"] for r in with_it) / len(with_it), 4)
                     if with_it else None},
            "without": {"games": len(without),
                        "win_rate": round(sum(r["won"] for r in without) / len(without), 4)
                        if without else None},
        }

    harbour_yes = [r for r in decided if r["opening_harbours"] > 0]
    harbour_no = [r for r in decided if r["opening_harbours"] == 0]

    return {
        "games": len(records),
        "decided": len(decided),
        "win_rate": round(sum(r["won"] for r in decided) / len(decided), 4),
        "mean_pips": round(sum(r["pips"] for r in decided) / len(decided), 4),
        "mean_gap_to_best": round(
            sum(r["gap_to_best"] for r in decided) / len(decided), 4),
        "mean_turns": round(sum(r["turns"] for r in decided) / len(decided), 1),
        "by_pips": bucket(decided, "pips",
                          quantile_edges([r["pips"] for r in decided]), "pips"),
        "by_diversity": [
            {"band": f"{d} resources",
             "games": len([r for r in decided if r["diversity"] == d]),
             "win_rate": round(
                 sum(r["won"] for r in decided if r["diversity"] == d)
                 / len([r for r in decided if r["diversity"] == d]), 4)}
            for d in sorted({r["diversity"] for r in decided})
            if len([r for r in decided if r["diversity"] == d])
        ],
        "by_strategy": bucket(decided, "ore_wheat_sheep",
                              quantile_edges([r["ore_wheat_sheep"] for r in decided]),
                              "OWS share"),
        "by_harbour_distance": bucket(
            decided, "gap_to_best",
            quantile_edges([r["gap_to_best"] for r in decided]), "pips forgone"),
        "by_resource": by_resource,
        "harbour_in_opening": {
            "with": {"games": len(harbour_yes),
                     "win_rate": round(
                         sum(r["won"] for r in harbour_yes) / len(harbour_yes), 4)
                     if harbour_yes else None},
            "without": {"games": len(harbour_no),
                        "win_rate": round(
                            sum(r["won"] for r in harbour_no) / len(harbour_no), 4)
                        if harbour_no else None},
        },
        "harbours_owned": round(
            sum(len(r["harbours_owned"]) for r in decided) / len(decided), 3),
        "top_spots": collections.Counter(
            v for r in decided for v in r["spots"]).most_common(12),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="What opening positions actually win",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--agent", default="champion",
                        help="'champion', 'heuristic', or a checkpoint path")
    parser.add_argument("--against", default="heuristic", choices=["heuristic", "greedy"])
    parser.add_argument("--out", default=str(STUDY))
    arguments = parser.parse_args(argv)

    import torch

    torch.set_num_threads(4)
    from catan.agents import GreedyAgent, HeuristicAgent

    if arguments.agent == "heuristic":
        agent = HeuristicAgent(1)
    elif arguments.agent == "champion":
        from training.alphazero.champion import load

        agent = load(simulations=arguments.simulations)
        if agent is None:
            parser.error("no AlphaZero champion to study")
    else:
        from training.alphazero.agent import MCTSAgent

        agent = MCTSAgent.load(arguments.agent, simulations=arguments.simulations)

    opponent = HeuristicAgent(0) if arguments.against == "heuristic" else GreedyAgent(0)

    print(f"{arguments.games} games, {arguments.agent} vs {arguments.against}")
    records = run(agent, opponent, games=arguments.games, seed=arguments.seed)
    summary = summarise(records)

    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "games": records}, indent=1),
                   encoding="utf-8")

    print(f"\nwin rate {100 * summary['win_rate']:.1f}% over {summary['decided']} decided")
    print(f"opening pips {summary['mean_pips']:.3f}, "
          f"left on the table {summary['mean_gap_to_best']:.3f}")
    print(f"harbours owned per game {summary['harbours_owned']:.2f}")
    print("\nwin rate by opening production:")
    for row in summary["by_pips"]:
        print(f"  {row['band']:<18} {100 * row['win_rate']:5.1f}%  ({row['games']} games)")
    print("\nwin rate by ore-wheat-sheep share of the opening:")
    for row in summary["by_strategy"]:
        print(f"  {row['band']:<18} {100 * row['win_rate']:5.1f}%  ({row['games']} games)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
