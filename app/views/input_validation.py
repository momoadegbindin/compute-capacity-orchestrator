from __future__ import annotations

import math


def validate_number(
    *,
    name: str,
    value: int | float,
    min_value: int | float,
    max_value: int | float,
    message: str | None = None,
    integer: bool = False,
) -> list[str]:
    """Validate a numeric input and return user-facing error messages."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return [message or f"{name} must be numeric."]

    if not math.isfinite(numeric_value):
        return [message or f"{name} must be finite."]

    if integer and not numeric_value.is_integer():
        return [message or f"{name} must be an integer."]

    if numeric_value < min_value or numeric_value > max_value:
        return [message or f"{name} must be between {min_value} and {max_value}."]

    return []


def validate_order(
    *,
    low_name: str,
    low_value: int | float,
    high_name: str,
    high_value: int | float,
    message: str | None = None,
) -> list[str]:
    """Validate that a lower-bound value does not exceed an upper-bound value."""
    try:
        numeric_low = float(low_value)
        numeric_high = float(high_value)
    except (TypeError, ValueError):
        return [message or f"{low_name} and {high_name} must be numeric."]

    if not math.isfinite(numeric_low) or not math.isfinite(numeric_high):
        return [message or f"{low_name} and {high_name} must be finite."]

    if numeric_low > numeric_high:
        return [message or f"{low_name} must be less than or equal to {high_name}."]

    return []