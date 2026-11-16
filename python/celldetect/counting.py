"""Counting statistics: what a sequencer actually hands you.

Used by: Weeks 1-2 (and by every module downstream).

Build order (each function may use the previous ones):
    1. sample_counts        (W1)
    2. fano_factor          (W1)
    3. poisson_logpmf       (W2)
    4. poisson_pmf          (W2)
    5. downsample_counts    (W2)

The whole project rests on one coincidence: for a Poisson variable, the
variance equals the mean. Everything you write here is a way of measuring,
using, or breaking that fact.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.special import gammaln


def sample_counts(lam: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_samples` independent Poisson counts with mean `lam`.
    This is the model of a sequencer: a gene expressed at rate `lam`
    molecules per cell yields a count that is Poisson(lam), and a different
    cell of the same type yields a different count. See Week 1.

    Parameters
    ----------
    lam       : float, the expected count (non-negative).
    n_samples : int, how many independent counts to draw.
    rng       : a numpy Generator, e.g. np.random.default_rng(101).
                It is a REQUIRED argument, not an optional one, so that no
                figure in this project can be produced without a seed.

    Returns
    -------
    numpy array of shape (n_samples,), integer dtype, all entries >= 0.

    Raises
    ------
    ValueError if lam < 0.

    Example
    -------
    >>> sample_counts(3.0, 5, np.random.default_rng(0))
    array([...])
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    return rng.poisson(lam, size=n_samples)
    raise NotImplementedError("Implement me! See docs/s/week01.html — and try before peeking at python/solutions/.")


def fano_factor(samples: Sequence[float]) -> float:
    """Return variance / mean of `samples` — 1.0 for a Poisson process.

    The Fano factor is the single most useful diagnostic in the project:
    exactly 1 for Poisson, above 1 for anything with extra (biological)
    variability. In Week 11 it is what tells you that real scRNA-seq counts
    are NOT Poisson, which is the project's most important negative result.

    Parameters
    ----------
    samples : 1-D sequence of counts.

    Returns
    -------
    float : var / mean, using the UNBIASED (ddof=1) variance.

    Raises
    ------
    ValueError if the sample mean is zero (the ratio is undefined).
    """
    np.asarray(samples, dtype=float)
    mean = np.mean(samples)
    if mean == 0:
        raise ValueError("Sample mean is zero; Fano factor undefined.")
    variance = np.var(samples, ddof=1)
    return float(variance / mean)
    raise NotImplementedError("Implement me! See docs/s/week01.html — and try before peeking at python/solutions/.")


def poisson_logpmf(k, lam: float) -> np.ndarray:
    """log P(X = k) for X ~ Poisson(lam), computed in log space.

    The formula (Week 2):
        log P(X = k) = k*log(lam) - lam - log(k!)
    and log(k!) = gammaln(k+1), already imported at the top of this file.

    Working in logs is not fussiness: at depth 5000 the factorial k!
    overflows a float64 (which caps out around 1e308) long before the
    probability itself becomes small. Compute the log, then exponentiate —
    never the other way round.

    Parameters
    ----------
    k   : int or array of ints (the counts to evaluate).
    lam : float, the Poisson mean (non-negative).

    Returns
    -------
    numpy array of log-probabilities, same shape as k.
    """
    k=np.asarray(k)
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if lam == 0:
        return np.where(k == 0, 0.0, -np.inf)
    return k * np.log(lam) - lam - gammaln(k + 1)
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def poisson_pmf(k, lam: float) -> np.ndarray:
    """P(X = k) for X ~ Poisson(lam).

    One line once poisson_logpmf works. Kept as its own function because
    the exact-error calculation in Week 4 wants probabilities, while the
    classifier in Week 5 wants logs.
    """
    return np.exp(poisson_logpmf(k, lam))
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def downsample_counts(counts, keep_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Thin counts: keep each molecule independently with probability `keep_prob`.

    This one function models BOTH of the project's "less data" knobs:
      * sequencing a library less deeply, and
      * the bead experiment's dropout rule.

    The key fact (Week 2, and you will prove it): if X ~ Poisson(lam) and
    each of the X items survives independently with probability q, the
    survivor count is Poisson(q*lam) — not approximately, exactly. Thinned
    Poisson data is still Poisson data, which is why the entire theory
    survives downsampling and why Week 11's real-data protocol is legitimate.

    Parameters
    ----------
    counts    : int or array of ints (works on a whole cells x genes matrix).
    keep_prob : float in [0, 1].
    rng       : numpy Generator.

    Returns
    -------
    numpy array of the same shape as `counts`, with every entry <= the input.

    Raises
    ------
    ValueError if keep_prob is outside [0, 1], or any count is negative.
    """
    if not (0 <= keep_prob <= 1):
        raise ValueError(f"keep_prob must be in [0, 1], got {keep_prob}")
    counts = np.asarray(counts)
    if np.any(counts < 0):
        raise ValueError("All counts must be non-negative.")
    return rng.binomial(counts.astype(np.int64), keep_prob)
    raise NotImplementedError("Implement me! See docs/s/week02.html — and try before peeking at python/solutions/.")


def total_counts(matrix) -> np.ndarray:
    """[PROVIDED] Library size per cell: the row sums of a cells x genes matrix."""
    return np.asarray(matrix).sum(axis=1)


def empirical_moments(samples: Sequence[float]) -> Tuple[float, float]:
    """[PROVIDED] Return (mean, unbiased variance) of a 1-D sample."""
    x = np.asarray(samples, dtype=float)
    return float(x.mean()), float(x.var(ddof=1))
