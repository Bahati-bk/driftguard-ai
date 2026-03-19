import numpy as np
from scipy.stats import ks_2samp


def psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    return np.sum((expected_counts - actual_counts) * np.log(expected_counts / actual_counts))


def ks_test(expected, actual):
    return ks_2samp(expected, actual)