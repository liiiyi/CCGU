"""Small numeric helpers shared by the mapping stages.

``scipy.stats.mode`` changed its return shape across the versions this project
can be installed against: before SciPy 1.9 it always returned length-1 arrays,
1.9-1.10 accept ``keepdims`` and warn when it is omitted, and from 1.11 the
default is ``keepdims=False``.  The mapping code calls it once per community, so
a single wrapper keeps Equation (9) identical on every supported SciPy.
"""

import numpy as np


def majority_label(values):
    """Return the most frequent value in ``values`` as a Python int.

    Ties resolve to the smallest value, matching ``scipy.stats.mode``.
    """
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        raise ValueError("majority_label() needs at least one value")
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])
