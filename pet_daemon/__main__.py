import argparse
import asyncio
import logging
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from .config import Config
from .daemon import Daemon
from .telemetry import ConsoleFormatter, configure


async def run(config):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY before starting pet-daemon")
    async with AsyncAnthropic(api_key=key, timeout=config.request_timeout, max_retries=0) as client:
        await Daemon(config, client).run()


def main():
    parser = argparse.ArgumentParser(prog="pet-daemon", description="Track coding-agent work and run a native Haiku desktop pet")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "pet.config.json")
    parser.add_argument("--verbose", action="store_true", help="Show debug events on stderr; JSONL always includes debug events")
    args = parser.parse_args()
    console = logging.StreamHandler()
    console.setFormatter(ConsoleFormatter(color=console.stream.isatty() and "NO_COLOR" not in os.environ))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, handlers=[console])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    try:
        config = Config.load(args.config)
        configure(config.log_file, args.verbose)
        asyncio.run(run(config))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        logging.getLogger("pet-daemon").exception("Daemon failed")
        parser.exit(1, f"pet-daemon: {error}\n")
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
