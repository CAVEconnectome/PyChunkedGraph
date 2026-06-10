"""Process exit codes mapped to the k8s Job podFailurePolicy.

FATAL is matched by an ``onExitCodes`` ``FailIndex`` rule so a known-bad chunk
fails its index immediately instead of burning the retry budget; TRANSIENT is an
ordinary failure that counts toward ``backoffLimitPerIndex``. Preemption needs no
code here — the kubelet stamps the ``DisruptionTarget`` condition and the
``Ignore`` rule keeps it off the budget.
"""

SUCCESS = 0
TRANSIENT = 1
FATAL = 42


class FatalChunkError(Exception):
    """Raised for a non-transient chunk failure (bad input/bug, not infra)."""
