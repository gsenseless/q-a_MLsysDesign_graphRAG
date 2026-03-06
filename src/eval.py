"""Evaluation script entry point.

This script generates test questions and evaluates the agent's responses.
"""
from pathlib import Path

from evaluation import generate_logs, evaluate_existing_logs


async def main() -> None:
    log_dir = Path("logs")
    await generate_logs(log_dir)
    await evaluate_existing_logs(log_dir)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
