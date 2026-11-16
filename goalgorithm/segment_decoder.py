"""Segment-aware event-modulated 3-state Viterbi decoder for H, A, D possession inference.

Objective function maximized over state sequence s_{0:T-1} in {0: H, 1: A, 2: D}:

    J(s) = sum_{t=0}^{T-1} E_t(s_t) - sum_{t=1}^{T-1} C_t(s_{t-1}, s_t)

where:
- E_t(H) and E_t(A) combine directional control, pass-flight continuity, uncertainty-gated
  memory, and bounded tactical priors, discounted by positive stoppage evidence.
- E_t(D) represents net stoppage vs live-play evidence.
- C_t(a, b) modulates base switch costs using bounded physical boundary-snapping cues:
  * H <-> A decreases near control acquisitions (bounded >= 0.45 * switch_cost) and
    increases slightly during contested duels or uninformative occlusions.
  * H/A -> D decreases near physical stoppage/exit boundaries (stop_edge).
  * D -> H/A decreases near physical restart-kick instants (restart_edge).
"""

import numpy as np


def build_transition_costs(
    n,
    switch_cost,
    dead_enter_cost,
    dead_exit_cost,
    acquisition_edge=None,
    contest_factor=None,
    stop_edge=None,
    restart_edge=None,
    occluded=None,
):
    """Compute time-varying (N-1, 3, 3) transition penalty tensor with bounded modulation."""
    if n < 2:
        return np.zeros((0, 3, 3), dtype=float)
    m = n - 1
    acq = np.zeros(m, dtype=float) if acquisition_edge is None else np.clip(acquisition_edge[1:], 0.0, 1.0)
    con = np.zeros(m, dtype=float) if contest_factor is None else np.clip(contest_factor[1:], 0.0, 1.0)
    stp = np.zeros(m, dtype=float) if stop_edge is None else np.clip(stop_edge[1:], 0.0, 1.0)
    rst = np.zeros(m, dtype=float) if restart_edge is None else np.clip(restart_edge[1:], 0.0, 1.0)
    occ = np.zeros(m, dtype=float) if occluded is None else np.clip(occluded[1:], 0.0, 1.0)

    c_ha = switch_cost * (1.0 - 0.45 * acq) * (1.0 + 0.35 * con + 0.20 * occ)
    c_ha = np.clip(c_ha, 0.45 * switch_cost, 1.55 * switch_cost)
    c_to_d = np.clip(dead_enter_cost * (1.0 - 0.55 * stp), 0.35 * dead_enter_cost, dead_enter_cost)
    c_from_d = np.clip(dead_exit_cost * (1.0 - 0.65 * rst), 0.30 * dead_exit_cost, dead_exit_cost)

    costs = np.zeros((m, 3, 3), dtype=float)
    costs[:, 0, 1] = c_ha
    costs[:, 1, 0] = c_ha
    costs[:, 0, 2] = c_to_d
    costs[:, 1, 2] = c_to_d
    costs[:, 2, 0] = c_from_d
    costs[:, 2, 1] = c_from_d
    return costs


def decode_segments(
    emissions,
    switch_cost,
    dead_enter_cost,
    dead_exit_cost=None,
    acquisition_edge=None,
    contest_factor=None,
    stop_edge=None,
    restart_edge=None,
    occluded=None,
):
    """Exact max-sum DP decoder with time-varying event-modulated transition penalties."""
    emissions = np.asarray(emissions, dtype=float)
    n = len(emissions)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    if dead_exit_cost is None:
        dead_exit_cost = dead_enter_cost

    costs = build_transition_costs(
        n,
        switch_cost,
        dead_enter_cost,
        dead_exit_cost,
        acquisition_edge=acquisition_edge,
        contest_factor=contest_factor,
        stop_edge=stop_edge,
        restart_edge=restart_edge,
        occluded=occluded,
    )

    e0 = emissions[:, 0].tolist()
    e1 = emissions[:, 1].tolist()
    e2 = emissions[:, 2].tolist()
    c_ha = costs[:, 0, 1].tolist()
    c_td = costs[:, 0, 2].tolist()
    c_fd = costs[:, 2, 0].tolist()

    back = np.empty((n, 3), dtype=np.int8)
    back[0] = 0
    s0, s1, s2 = e0[0], e1[0], e2[0]

    for i in range(1, n):
        cha = c_ha[i - 1]
        ctd = c_td[i - 1]
        cfd = c_fd[i - 1]

        # Best predecessor for state 0 (H)
        v00 = s0
        v10 = s1 - cha
        v20 = s2 - cfd
        if v00 >= v10 and v00 >= v20:
            b0, n0 = 0, v00 + e0[i]
        elif v10 >= v20:
            b0, n0 = 1, v10 + e0[i]
        else:
            b0, n0 = 2, v20 + e0[i]

        # Best predecessor for state 1 (A)
        v01 = s0 - cha
        v11 = s1
        v21 = s2 - cfd
        if v01 >= v11 and v01 >= v21:
            b1, n1 = 0, v01 + e1[i]
        elif v11 >= v21:
            b1, n1 = 1, v11 + e1[i]
        else:
            b1, n1 = 2, v21 + e1[i]

        # Best predecessor for state 2 (D)
        v02 = s0 - ctd
        v12 = s1 - ctd
        v22 = s2
        if v02 >= v12 and v02 >= v22:
            b2, n2 = 0, v02 + e2[i]
        elif v12 >= v22:
            b2, n2 = 1, v12 + e2[i]
        else:
            b2, n2 = 2, v22 + e2[i]

        m_val = n0 if n0 >= n1 and n0 >= n2 else (n1 if n1 >= n2 else n2)
        s0, s1, s2 = n0 - m_val, n1 - m_val, n2 - m_val
        back[i, 0] = b0
        back[i, 1] = b1
        back[i, 2] = b2

    states = np.empty(n, dtype=np.int8)
    last = 0 if s0 >= s1 and s0 >= s2 else (1 if s1 >= s2 else 2)
    states[-1] = last
    cur = last
    for i in range(n - 1, 0, -1):
        cur = int(back[i, cur])
        states[i - 1] = cur
    return states
