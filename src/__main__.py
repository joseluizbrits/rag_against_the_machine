"""Main module for RAG against the machine."""

import sys

import fire

from src.cli import RagCli


def main() -> None:
    """Run the CLI, turning validation/I/O errors into a one-line message."""
    try:
        fire.Fire(RagCli())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
