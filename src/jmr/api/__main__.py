"""Run the single-worker JMR FastAPI server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("JMR_API_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("JMR_API_PORT", "8000"))
    )
    args = parser.parse_args(argv)
    uvicorn.run(
        "jmr.api.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
