"""CLI argument parsing and command handling for omanage."""

# Import main entry point from parser module
from .cli_core.parser import main, create_parser

# Re-export for backward compatibility
__all__ = ['main', 'create_parser']