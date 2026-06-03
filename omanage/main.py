#!/usr/bin/env python3
"""Main entry point for the omanage CLI tool."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())