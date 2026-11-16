"""Three-state Viterbi with evidence-dependent change costs and half-local memory."""

import numpy as np


def controller_memory(evidence, reliability, fps, seconds):
    """Forward memory preserves the last controlled team in passes and occlusions."""
    out = np.zeros(len(evidence))
    memory = 0.
    decay = np.exp(-1/(fps*seconds))
    for i, value in enumerate(evidence):
        memory *= decay
        if reliability[i] > 0.2 and abs(value) > 0.15:
            memory = 0.8*memory + 0.2*value
        out[i] = memory
    return out


def viterbi(emissions, switch_cost, dead_cost, recovery=None):
    """Max-sum chain. Bounded scores are evidence, not calibrated log probabilities.

    Explicit transition costs impose an evidence-dependent duration preference without
    deleting every short possession. Costs reset at each independent call/half.
    """
    n = len(emissions)
    if not n:
        return np.empty(0, dtype=np.int8)
    penalty = np.array([[0., switch_cost, dead_cost], [switch_cost, 0., dead_cost],
                        [dead_cost, dead_cost, 0.]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emissions[0].copy()
    for i in range(1, n):
        costs = penalty.copy()
        if recovery is not None:
            # Strong control makes a transition cheaper but never free.
            costs[0, 1] *= 1-0.25*recovery[i]
            costs[1, 0] *= 1-0.25*recovery[i]
        candidates = score[:, None]-costs
        back[i] = np.argmax(candidates, axis=0)
        score = candidates[back[i], np.arange(3)] + emissions[i]
        score -= score.max()
    states = np.empty(n, dtype=np.int8)
    states[-1] = np.argmax(score)
    for i in range(n-1, 0, -1):
        states[i-1] = back[i, states[i]]
    return states
