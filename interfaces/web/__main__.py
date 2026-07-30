"""``python -m interfaces.web`` — start the local game server."""

import argparse

from interfaces.web.server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m interfaces.web",
        description="Play Catan against the AI in a browser.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="localhost by default — there is no authentication")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
