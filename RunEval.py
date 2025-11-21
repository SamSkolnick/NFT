import sys
from typing import Optional, Sequence

from green_agent_cli import ensure_python310, main as cli_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Backwards-compatible entrypoint that delegates to the unified CLI.

    Running `python RunEval.py --config ...` behaves the same as invoking
    `python -m green_agent_cli serve --config ...`.
    """

    ensure_python310()

    if argv is None:
        argv = sys.argv[1:]

    return cli_main(["serve", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
