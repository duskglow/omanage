"""Secure subprocess utilities for omanage - model name validation and command execution."""

import subprocess
from typing import List, Optional, Any

from ..utils import validate_model_name, InvalidModelNameError


# Default timeout for subprocess calls (seconds)
DEFAULT_TIMEOUT = 300


class SubprocessError(Exception):
    """Custom exception for subprocess execution errors."""
    pass


def run_ollama_command(
    cmd: List[str],
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
    ollama_binary: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run an Ollama command with proper input validation.
    
    This function provides a centralized, secure way to execute Ollama commands
    with consistent model name validation to prevent command injection.
    
    Args:
        cmd: The Ollama command to execute as a list of arguments.
        model_name: Optional model name to validate before execution.
                   If provided, validates the model name before running the command.
        timeout: Timeout in seconds for the subprocess call. Defaults to
                DEFAULT_TIMEOUT (300 seconds).
        ollama_binary: Optional path to the Ollama binary. If provided, replaces
                      the first element of cmd with this binary path.
    
    Returns:
        subprocess.CompletedProcess with the command result.
    
    Raises:
        SubprocessError: If command execution fails or validation fails.
        InvalidModelNameError: If model name contains invalid characters.
    
    Example:
        >>> # No model name validation
        >>> result = run_ollama_command(["ollama", "list"])
        >>> 
        >>> # With model name validation
        >>> result = run_ollama_command(["ollama", "show", "--modelfile", "my-model:latest"], 
        ...                             model_name="my-model:latest")
    """
    # Validate model name if provided
    if model_name:
        try:
            validate_model_name(model_name)
        except InvalidModelNameError as e:
            raise SubprocessError(f"Invalid model name for command: {e}")
        
        # Additional length validation to prevent buffer overflow attacks
        if len(model_name) > 256:
            raise SubprocessError(f"Model name too long (max 256 characters): {model_name}")
    
    # Apply custom binary if specified
    if ollama_binary and cmd:
        # Validate the binary path exists and is executable to prevent
        # arbitrary command execution via a malicious ollama_binary value.
        import shutil
        resolved = shutil.which(ollama_binary)
        if not resolved:
            raise SubprocessError(
                f"Ollama binary not found or not executable: {ollama_binary}"
            )
        cmd = [resolved] + cmd[1:]
    
    # Validate command arguments are safe strings
    for arg in cmd:
        if not isinstance(arg, str):
            raise SubprocessError(f"Command argument must be a string: {arg}")
        # Prevent command injection via argument length
        if len(arg) > 1024:
            raise SubprocessError(f"Command argument too long: {arg[:50]}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout or DEFAULT_TIMEOUT
        )
        return result
    except subprocess.CalledProcessError as e:
        raise SubprocessError(
            f"Ollama command failed: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"Error: {e.stderr.strip() if e.stderr else 'No error output'}"
        )
    except FileNotFoundError:
        raise SubprocessError(
            "Ollama not found. Please install Ollama first and ensure it's in your PATH."
        )
    except subprocess.TimeoutExpired:
        raise SubprocessError(
            f"Ollama command timed out after {timeout or DEFAULT_TIMEOUT} seconds: {' '.join(cmd)}"
        )


def get_ollama_models(ollama_binary: Optional[str] = None) -> List[dict]:
    """
    Get list of installed Ollama models securely.
    
    This is a centralized function that uses the secure subprocess wrapper
    to get the list of installed models.
    
    Args:
        ollama_binary: Optional path to the Ollama binary to use.
    
    Returns:
        List of dictionaries with 'name' key for each installed model.
    
    Raises:
        SubprocessError: If the Ollama command fails.
    """
    result = run_ollama_command(["ollama", "list"], ollama_binary=ollama_binary)
    
    lines = result.stdout.strip().split('\n')
    models = []
    
    for line in lines[1:]:  # Skip header line
        if line.strip():
            parts = line.split()
            if parts:
                # Validate model name before adding
                model_name = parts[0]
                try:
                    validate_model_name(model_name)
                    models.append({"name": model_name})
                except InvalidModelNameError:
                    # Skip invalid model names
                    continue
    
    return models


def get_model_blob_info(model_name: str, ollama_binary: Optional[str] = None) -> Optional[dict]:
    """
    Get blob information for a model from its modelfile securely.
    
    This is a centralized function that uses the secure subprocess wrapper
    to get blob information from the Ollama modelfile.
    
    Args:
        model_name: Name of the model to query.
        ollama_binary: Optional path to the Ollama binary to use.
    
    Returns:
        Dictionary with 'blobSha' and 'blobName' keys, or None if modelfile
        does not contain a FROM line.
    
    Raises:
        SubprocessError: If the Ollama command fails or Ollama is not installed.
        InvalidModelNameError: If model name is invalid.
    """
    # Validate model name first
    try:
        validate_model_name(model_name)
    except InvalidModelNameError as e:
        raise SubprocessError(f"Cannot get blob info: {e}")
    
    result = run_ollama_command(
        ["ollama", "show", "--modelfile", model_name],
        model_name=model_name,
        ollama_binary=ollama_binary
    )
    
    for line in result.stdout.split('\n'):
        if line.startswith("FROM "):
            from_path = line[5:].strip()
            # Use Path.name so the extraction works on both Unix and Windows paths
            blob_name = Path(from_path).name
            return {
                "blobSha": blob_name,  # Ollama uses SHA256 digest as blob identifier/filename
                "blobName": blob_name  # Full blob filename (same as SHA in Ollama's convention)
            }
    
    return None
