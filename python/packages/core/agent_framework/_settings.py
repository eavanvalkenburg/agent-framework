# Copyright (c) Microsoft. All rights reserved.

"""Generic settings loader with environment variable resolution.

This module provides a ``load_settings()`` function that populates a ``TypedDict``
from environment variables, ``.env`` files, and explicit overrides.  It replaces
the previous pydantic-settings-based ``AFBaseSettings`` with a lighter-weight,
function-based approach that has no pydantic-settings dependency.

Usage::

    class MySettings(TypedDict, total=False):
        api_key: str | None  # optional — resolves to None if not set
        model_id: str | None  # optional by default
        source_a: str | None
        source_b: str | None


    # Make model_id required; require exactly one of source_a / source_b:
    settings = load_settings(
        MySettings,
        env_prefix="MY_APP_",
        required_fields=["model_id", ("source_a", "source_b")],
        model_id="gpt-4",
        source_a="value",
    )
    settings["api_key"]  # type-checked dict access
    settings["model_id"]  # str | None per type, but guaranteed not None at runtime
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, TypedDict, Union, cast, get_args, get_origin, get_type_hints

from dotenv import dotenv_values

from .exceptions import SettingNotFoundError

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypeVar  # type: ignore # pragma: no cover


SettingsT = TypeVar("SettingsT", default=dict[str, Any])
BackendT = TypeVar("BackendT", bound=str)


class _BackendSetting(TypedDict, total=False):
    """Internal backend settings shape for generic backend resolution."""

    backend: str | None


class SecretString(str):
    """A string subclass that masks its value in repr() to prevent accidental exposure.

    SecretString behaves exactly like a regular string in all operations,
    but its repr() shows '**********' instead of the actual value.
    This helps prevent secrets from being accidentally logged or displayed.

    It also provides a ``get_secret_value()`` method for backward compatibility
    with code that previously used ``pydantic.SecretStr``.

    Example:
        ```python
        api_key = SecretString("sk-secret-key")
        print(api_key)  # sk-secret-key (normal string behavior)
        print(repr(api_key))  # SecretString('**********')
        print(f"Key: {api_key}")  # Key: sk-secret-key
        print(api_key.get_secret_value())  # sk-secret-key
        ```
    """

    def __repr__(self) -> str:
        """Return a masked representation to prevent secret exposure."""
        return "SecretString('**********')"

    def get_secret_value(self) -> str:
        """Return the underlying string value.

        Provided for backward compatibility with ``pydantic.SecretStr``.
        Since SecretString *is* a str, this simply returns ``str(self)``.
        """
        return str(self)


def _coerce_value(value: str, target_type: type) -> Any:
    """Coerce a string value to the target type."""
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Handle Union types (e.g., str | None) — try each non-None arm
    if origin is type(None):
        return None

    if args and type(None) in args:
        for arg in args:
            if arg is not type(None):
                with suppress(ValueError, TypeError):
                    return _coerce_value(value, arg)
        return value

    # Handle SecretString
    if target_type is SecretString or (isinstance(target_type, type) and issubclass(target_type, SecretString)):
        return SecretString(value)

    # Handle basic types
    if target_type is str:
        return value
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is bool:
        return value.lower() in ("true", "1", "yes", "on")

    return value


def _check_override_type(value: Any, field_type: type, field_name: str) -> None:
    """Validate that *value* is compatible with *field_type*.

    Raises ``ValueError`` when the override is clearly
    incompatible (e.g. a ``dict`` passed where ``str`` is expected).
    Callable values and ``None`` are always accepted.
    """
    if value is None:
        return

    # Callables are always allowed (e.g. lazy token providers)
    if callable(value) and not isinstance(value, (str, bytes)):
        return

    # Collect the concrete types that *field_type* allows
    origin = get_origin(field_type)
    args = get_args(field_type)

    allowed: tuple[type, ...]
    if origin is Union or origin is type(int | str):
        allowed = tuple(a for a in args if isinstance(a, type) and a is not type(None))
        # If any arm is a Callable, allow anything callable
        if any(get_origin(a) is Callable or a is Callable for a in args):
            return
    elif isinstance(field_type, type):
        allowed = (field_type,)
    else:
        return  # complex / unknown annotation — skip check

    if not allowed:
        return

    if not isinstance(value, allowed):
        # Allow str for SecretString fields (will be coerced)
        if isinstance(value, str) and any(isinstance(a, type) and issubclass(a, str) for a in allowed):
            return
        # Allow int for float fields (standard numeric promotion)
        if isinstance(value, int) and float in allowed:
            return

        allowed_names = ", ".join(t.__name__ for t in allowed)
        raise ValueError(
            f"Invalid type for setting '{field_name}': expected {allowed_names}, got {type(value).__name__}."
        )


def _get_env_var_names(
    field_name: str,
    env_prefix: str,
    env_var_names: Mapping[str, str | Sequence[str]] | None,
) -> tuple[str, ...]:
    """Return the environment variable names to check for a field."""
    if env_var_names is None or field_name not in env_var_names:
        return (f"{env_prefix}{field_name.upper()}",)

    configured_names = env_var_names[field_name]
    if isinstance(configured_names, str):
        return (configured_names,)

    normalized_names = tuple(dict.fromkeys(configured_names))
    if normalized_names:
        return normalized_names
    return (f"{env_prefix}{field_name.upper()}",)


def _format_env_var_hint(env_names: Sequence[str]) -> str:
    """Format environment variable names for error messages."""
    if len(env_names) == 1:
        return f"the '{env_names[0]}' environment variable"

    env_list = ", ".join(f"'{env_name}'" for env_name in env_names)
    return f"one of the {env_list} environment variables"


def resolve_backend_setting(
    *,
    backend: BackendT | None,
    env_var_name: str,
    default_backend: BackendT,
    allowed_backends: tuple[BackendT, ...],
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> BackendT:
    """Resolve a backend selector from explicit input or environment state.

    Args:
        backend: Explicit backend override when provided.
        env_var_name: Environment variable name used to resolve the backend when no explicit
            value is supplied.
        default_backend: Backend to use when no explicit or environment value is set.
        allowed_backends: Allowed backend values.
        env_file_path: Optional `.env` file to read before process environment variables.
        env_file_encoding: Encoding for the `.env` file.

    Returns:
        The resolved backend value.

    Raises:
        ValueError: If the resolved backend is not one of `allowed_backends`.
    """
    settings = load_settings(
        _BackendSetting,
        backend=backend,
        env_var_names={"backend": (env_var_name,)},
        env_file_path=env_file_path,
        env_file_encoding=env_file_encoding,
    )
    resolved_backend = cast(BackendT | None, settings.get("backend"))
    if resolved_backend is None:
        return default_backend
    if resolved_backend not in allowed_backends:
        allowed = ", ".join(f"'{backend_name}'" for backend_name in allowed_backends)
        raise ValueError(f"Unsupported backend {resolved_backend!r}. Expected one of: {allowed}.")
    return resolved_backend


def load_settings(
    settings_type: type[SettingsT],
    *,
    env_prefix: str = "",
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    env_var_names: Mapping[str, str | Sequence[str]] | None = None,
    required_fields: Sequence[str | tuple[str, ...]] | None = None,
    **overrides: Any,
) -> SettingsT:
    """Load settings from explicit overrides, an optional ``.env`` file, and environment variables.

    The *settings_type* must be a ``TypedDict`` subclass.  Values are resolved in
    this order (highest priority first):

    1. Explicit keyword *overrides* (``None`` values are filtered out).
    2. A ``.env`` file (when *env_file_path* is explicitly provided).
    3. Environment variables (``<env_prefix><FIELD_NAME>``).
    4. Default values — fields with class-level defaults on the TypedDict, or
       ``None`` for optional fields.

    Entries in *required_fields* are validated after resolution:

    - A **string** entry means the field must resolve to a non-``None`` value.
    - A **tuple** entry means exactly one field in the group must be non-``None``
      (mutually exclusive).

    Args:
        settings_type: A ``TypedDict`` class describing the settings schema.
        env_prefix: Prefix for environment variable lookup (e.g. ``"OPENAI_"``).
        env_file_path: Path to ``.env`` file. When provided, the file is required
            and values are resolved before process environment variables.
        env_file_encoding: Encoding for reading the ``.env`` file.  Defaults to ``"utf-8"``.
        env_var_names: Optional mapping from setting field names to explicit environment
            variable names to check, in priority order. When provided for a field, these names
            are used instead of the default ``<env_prefix><FIELD_NAME>`` lookup.
        required_fields: Field names (``str``) that must resolve to a non-``None``
            value, or tuples of field names where exactly one must be set.
        **overrides: Field values.  ``None`` values are ignored so that callers can
            forward optional parameters without masking env-var / default resolution.

    Returns:
        A populated dict matching *settings_type*.

    Raises:
        FileNotFoundError: If *env_file_path* was provided but the file does not exist.
        SettingNotFoundError: If a required field could not be resolved from any
            source, or if a mutually exclusive constraint is violated.
        ValueError: If an override value has an incompatible type.
    """
    encoding = env_file_encoding or "utf-8"

    loaded_dotenv_values: dict[str, str] = {}
    if env_file_path is not None:
        if not os.path.exists(env_file_path):
            raise FileNotFoundError(env_file_path)

        raw_dotenv_values = dotenv_values(dotenv_path=env_file_path, encoding=encoding)
        loaded_dotenv_values = {key: value for key, value in raw_dotenv_values.items() if value is not None}

    # Filter out None overrides so defaults / env vars are preserved
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # Get field type hints from the TypedDict
    hints = get_type_hints(settings_type)

    result: dict[str, Any] = {}
    for field_name, field_type in hints.items():
        # 1. Explicit override wins
        if field_name in overrides:
            override_value = overrides[field_name]
            _check_override_type(override_value, field_type, field_name)
            # Coerce plain str → SecretString if the annotation expects it
            if isinstance(override_value, str) and not isinstance(override_value, SecretString):
                with suppress(ValueError, TypeError):
                    coerced = _coerce_value(override_value, field_type)
                    if isinstance(coerced, SecretString):
                        override_value = coerced
            result[field_name] = override_value
            continue

        env_names = _get_env_var_names(field_name, env_prefix, env_var_names)

        # 2. Optional .env value (only when env_file_path is explicitly provided)
        if loaded_dotenv_values:
            for env_var_name in env_names:
                dotenv_value = loaded_dotenv_values.get(env_var_name)
                if dotenv_value is None:
                    continue
                try:
                    result[field_name] = _coerce_value(dotenv_value, field_type)
                except (ValueError, TypeError):
                    result[field_name] = dotenv_value
                break
            if field_name in result:
                continue

        # 3. Environment variable
        for env_var_name in env_names:
            env_value = os.getenv(env_var_name)
            if env_value is None:
                continue
            try:
                result[field_name] = _coerce_value(env_value, field_type)
            except (ValueError, TypeError):
                result[field_name] = env_value
            break
        if field_name in result:
            continue

        # 4. Default from TypedDict class-level defaults, or None for optional fields
        if hasattr(settings_type, field_name):
            result[field_name] = getattr(settings_type, field_name)
        else:
            result[field_name] = None

    # Validate required fields after all resolution
    if required_fields:
        for entry in required_fields:
            if isinstance(entry, str):
                # Single required field
                if result.get(entry) is None:
                    env_names = _get_env_var_names(entry, env_prefix, env_var_names)
                    raise SettingNotFoundError(
                        f"Required setting '{entry}' was not provided. "
                        f"Set it via the '{entry}' parameter or the "
                        f"{_format_env_var_hint(env_names)}."
                    )
            else:
                # Mutually exclusive group — exactly one must be set
                set_fields = [f for f in entry if result.get(f) is not None]
                if len(set_fields) == 0:
                    names = ", ".join(f"'{f}'" for f in entry)
                    raise SettingNotFoundError(f"Exactly one of {names} must be provided, but none was set.")
                if len(set_fields) > 1:
                    all_names = ", ".join(f"'{f}'" for f in entry)
                    set_names = ", ".join(f"'{f}'" for f in set_fields)
                    raise SettingNotFoundError(
                        f"Only one of {all_names} may be provided, but multiple were set: {set_names}."
                    )

    return result  # type: ignore[return-value]
