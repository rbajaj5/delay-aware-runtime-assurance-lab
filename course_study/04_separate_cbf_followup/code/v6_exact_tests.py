"""Exact paired-binomial helpers for V6 McNemar analyses.

The directional alternative is explicit because the left and right tails are
not interchangeable. Counts are the discordant cells from a matched-pair
table: ``left_only`` and ``right_only``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal


Alternative = Literal["left_fewer", "left_greater"]


def _validate_counts(left_only: int, right_only: int) -> None:
    if isinstance(left_only, bool) or isinstance(right_only, bool):
        raise TypeError("discordant counts must be integers, not booleans")
    if not isinstance(left_only, int) or not isinstance(right_only, int):
        raise TypeError("discordant counts must be integers")
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be nonnegative")


def exact_mcnemar_one_sided_fraction(
    left_only: int,
    right_only: int,
    *,
    alternative: Alternative,
) -> Fraction:
    """Return the exact directed McNemar p-value as a rational number.

    Under the null, the number of left-only outcomes among discordant pairs
    follows Binomial(n=left_only+right_only, p=1/2).
    """

    _validate_counts(left_only, right_only)
    discordant = left_only + right_only
    if discordant == 0:
        return Fraction(1, 1)

    if alternative == "left_fewer":
        numerator = sum(
            math.comb(discordant, index)
            for index in range(0, left_only + 1)
        )
    elif alternative == "left_greater":
        numerator = sum(
            math.comb(discordant, index)
            for index in range(left_only, discordant + 1)
        )
    else:
        raise ValueError(
            "alternative must be 'left_fewer' or 'left_greater'"
        )
    return Fraction(numerator, 2**discordant)


def exact_mcnemar_one_sided(
    left_only: int,
    right_only: int,
    *,
    alternative: Alternative,
) -> float:
    """Return the exact directed McNemar p-value as a float."""

    return float(
        exact_mcnemar_one_sided_fraction(
            left_only,
            right_only,
            alternative=alternative,
        )
    )


def exact_mcnemar_two_sided_fraction(
    left_only: int,
    right_only: int,
) -> Fraction:
    """Return the doubled-smaller-tail exact McNemar p-value."""

    _validate_counts(left_only, right_only)
    lower = exact_mcnemar_one_sided_fraction(
        left_only,
        right_only,
        alternative="left_fewer",
    )
    upper = exact_mcnemar_one_sided_fraction(
        left_only,
        right_only,
        alternative="left_greater",
    )
    return min(Fraction(1, 1), 2 * min(lower, upper))


def exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    """Return the doubled-smaller-tail exact McNemar p-value as a float."""

    return float(exact_mcnemar_two_sided_fraction(left_only, right_only))
