"""Where the time goes.

    python -m benchmark.profiler engine
    python -m benchmark.profiler selfplay --simulations 48

:mod:`benchmark.benchmark` says *how fast*; this says *why*. Section 8 of the guide names the
suspects — deep copies, legal move generation, resource distribution, longest road,
allocations — and this points cProfile at them so the answer is measured rather than guessed.

Two things worth knowing before reading a profile of this codebase.

**cProfile's overhead falls on small functions**, and this engine is made of them:
``legal_actions`` at 13 us and ``clone`` at 4 us are inside the range where the profiler's
per-call cost is comparable to the call. Ratios between lines are trustworthy; absolute
times are not. Use :mod:`benchmark.benchmark` for absolutes.

**The self-play profile is dominated by whatever the network is doing**, which will be one
opaque ``torch`` frame. That is the honest picture — at 48 simulations a move the network is
most of the cost — but it means the interesting Python is below it, so the default here sorts
by *own* time rather than cumulative.
"""

import argparse
import cProfile
import pstats


def profile(callable_, sort="tottime", limit=30, log=print):
    """Run ``callable_`` under cProfile and print the top ``limit`` rows."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = callable_()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(sort)
    log(f"\ntop {limit} by {sort}")
    stats.print_stats(limit)
    return result, stats


def engine(games=100, seed=0):
    """Profile the simulator with no network in the loop."""
    from benchmark.benchmark import play_random_game

    def run():
        for game in range(games):
            play_random_game(seed + game)

    return profile(run)


def selfplay(positions=600, simulations=48, width=16, seed=0, checkpoint=None):
    """Profile one worker's self-play: search, engine and network together."""
    import torch

    torch.set_num_threads(1)
    from training.alphazero.network import load_for_alphazero, new_network
    from training.alphazero.self_play import Generator

    net = load_for_alphazero(checkpoint)[0] if checkpoint else new_network()
    net.eval()

    def evaluate(obs, masks):
        with torch.no_grad():
            logits, value = net(torch.from_numpy(obs))
            logits = net._apply_mask(logits, torch.from_numpy(masks))
            return torch.softmax(logits, dim=-1).numpy(), value.numpy()

    generator = Generator(evaluate, {"simulations": simulations}, seed=seed, width=width)
    generator.run(positions=positions // 4)                  # warm-up, outside the profile

    return profile(lambda: generator.run(positions=positions))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Where the time goes",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("what", nargs="?", default="engine", choices=["engine", "selfplay"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--positions", type=int, default=600)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    arguments = parser.parse_args(argv)

    if arguments.what == "engine":
        engine(games=arguments.games)
    else:
        selfplay(positions=arguments.positions, simulations=arguments.simulations,
                 width=arguments.width, checkpoint=arguments.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
