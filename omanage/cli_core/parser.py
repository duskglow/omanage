"""Argument parser for omanage CLI."""

import argparse
import sys
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="omanage",
        description="Ollama Model Manager - Manage Ollama model storage"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show or set configuration options"
    )
    config_parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        help="Set a configuration key"
    )
    config_parser.add_argument(
        "--get",
        metavar="KEY",
        help="Get a configuration key"
    )
    config_parser.set_defaults(func=_get_command_handler('cmd_config'))
    
    # help command
    help_parser = subparsers.add_parser(
        "help",
        help="Show help message"
    )
    help_parser.set_defaults(func=_get_command_handler('cmd_help'))
    
    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all models with their status"
    )
    list_parser.set_defaults(func=_get_command_handler('cmd_list'))
    
    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize model index from Ollama"
    )
    init_parser.add_argument(
        "model_name",
        nargs="?",
        help="Specific model to initialize (optional)"
    )
    init_parser.set_defaults(func=_get_command_handler('cmd_init'))
    
    # refresh command
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Refresh model index from Ollama"
    )
    refresh_parser.add_argument(
        "model_name",
        nargs="?",
        help="Specific model to refresh (optional)"
    )
    refresh_parser.set_defaults(func=_get_command_handler('cmd_refresh'))
    
    # freeze command
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Move a model's blob to remote storage"
    )
    freeze_parser.add_argument(
        "model_name",
        help="Model to freeze"
    )
    freeze_parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the blob during move"
    )
    freeze_parser.set_defaults(func=_get_command_handler('cmd_freeze'))
    
    # thaw command
    thaw_parser = subparsers.add_parser(
        "thaw",
        help="Move a model's blob back to base storage"
    )
    thaw_parser.add_argument(
        "model_name",
        help="Model to thaw"
    )
    thaw_parser.set_defaults(func=_get_command_handler('cmd_thaw'))
    
    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify model file locations match index"
    )
    verify_parser.set_defaults(func=_get_command_handler('cmd_verify'))
    
    return parser


def _get_command_handler(name: str):
    """Get command handler function by name."""
    from .commands import cmd_config, cmd_help, cmd_list, cmd_init, cmd_refresh, cmd_freeze, cmd_thaw, cmd_verify
    
    handlers = {
        'cmd_config': cmd_config,
        'cmd_help': cmd_help,
        'cmd_list': cmd_list,
        'cmd_init': cmd_init,
        'cmd_refresh': cmd_refresh,
        'cmd_freeze': cmd_freeze,
        'cmd_thaw': cmd_thaw,
        'cmd_verify': cmd_verify,
    }
    return handlers[name]


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code if e.code is not None else 1
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Execute the command
    try:
        return parsed_args.func(parsed_args)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
