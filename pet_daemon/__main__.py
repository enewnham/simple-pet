import argparse
import asyncio
import logging
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from .config import Config
from .daemon import Daemon
from .logging import configure, get_logger


async def run(config: Config) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY before starting pet-daemon")
    async with AsyncAnthropic(
        api_key=key, timeout=config.request_timeout, max_retries=0
    ) as client:
        await Daemon(config, client).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pet-daemon",
        description="Track coding-agent work and run a native Haiku desktop pet",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pet.config.json",
    )
    detail = parser.add_mutually_exclusive_group()
    detail.add_argument(
        "--debug",
        action="store_true",
        help="Show debug events on stderr; JSONL includes all levels",
    )
    detail.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug events and full payloads on stderr; JSONL includes all levels",
    )
    args = parser.parse_args()
    configure(verbose=args.verbose, debug=args.debug)
    try:
        config = Config.load(args.config)
        configure(config.log_file, args.verbose, args.debug)
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass
    except OSError, ValueError, TypeError, KeyError, RuntimeError:
        get_logger("pet-daemon").exception("daemon.failed")
        parser.exit(1)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
