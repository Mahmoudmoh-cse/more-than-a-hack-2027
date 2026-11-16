"""Multi-frame ball-control reasoning, duel attenuation, pass continuity, and slot-agnostic tracking."""

import numpy as np

from .features import rolling_mean, runs, sustained


def _slot_agnostic_continuity(players, nearest_pos, slots, fps, config):
    """True where the nearest team player matches any teammate position on the prior frame."""
    n = len(players)
    out = np.zeros(n, dtype=bool)
    if n < 2 or len(slots) == 0:
        return out
    step_max = config.max_player_speed / fps + config.position_slack
    prev_team = players[:-1, slots]  # (N-1, M, 2)
    cur_near = nearest_pos[1:, None, :]  # (N-1, 1, 2)
    step_dist = np.linalg.norm(prev_team - cur_near, axis=2)
    step_dist[~np.isfinite(step_dist)] = np.inf
    out[1:] = np.min(step_dist, axis=1) < step_max
    return out


def compute_control_evidence(features, teams, fps, config):
    """Compute multi-frame team control evidence, duel attenuation, pass bridges, and gains."""
    players = features["players"]
    ball = features["ball"]
    usable = features["usable"]
    reliability = features["reliability"]
    ball_speed = features["ball_speed"]
    turn_sharpness = features["turn_sharpness"]
    player_vel = features["player_velocity"]
    ball_vel = features["ball_velocity"]
    dedup_valid = features["dedup_valid"]
    n = len(ball)
    lag = max(2, round(config.velocity_seconds * fps))

    diff = players - ball[:, None, :]
    distances = np.linalg.norm(diff, axis=2)
    distances[~np.isfinite(distances)] = np.inf
    # Use deduplicated slots when choosing representatives & densities.
    distances_dedup = np.where(dedup_valid, distances, np.inf)

    nearest = np.full((n, 2), np.inf)
    density = np.zeros((n, 2), dtype=float)
    relative = np.full((n, 2), np.nan)
    consistency = np.zeros((n, 2), dtype=float)
    support = np.zeros((n, 2), dtype=float)
    centroids = np.full((n, 2, 2), np.nan)
    spread = np.full((n, 2), np.nan)
    rows = np.arange(n)

    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if len(slots) == 0:
            continue
        ds = distances_dedup[:, slots]
        # Fallback to raw distances on frames where deduplication filtered out all slots.
        ds_raw = distances[:, slots]
        use_ds = np.where(np.any(np.isfinite(ds), axis=1)[:, None], ds, ds_raw)
        best_local = np.argmin(use_ds, axis=1)
        selected = slots[best_local]
        nearest[:, team] = use_ds[rows, best_local]
        density[:, team] = (use_ds < config.density_radius).sum(axis=1)
        relative[:, team] = np.linalg.norm(player_vel[rows, selected] - ball_vel, axis=1)

        nearest_pos = players[rows, selected]
        cont = _slot_agnostic_continuity(players, nearest_pos, slots, fps, config)
        consistency[:, team] = rolling_mean(cont, lag)

        prox = np.exp(-nearest[:, team] / config.control_radius)
        co_motion = np.where(
            np.isfinite(relative[:, team]),
            np.exp(-relative[:, team] / 5.5),
            0.35,
        )
        raw = prox * (0.60 + 0.25 * co_motion + 0.15 * consistency[:, team])

        # Attenuate bystanders standing near a fast-moving pass in flight.
        bystander = (
            np.isfinite(ball_speed)
            & (ball_speed > 9.0)
            & (nearest[:, team] > 1.7)
            & (~np.isfinite(relative[:, team]) | (relative[:, team] > 6.5))
        )
        bystander_scale = np.where(
            bystander,
            1.0 / (1.0 + ((np.nan_to_num(ball_speed, nan=9.0) - 9.0) / 8.0) ** 2),
            1.0,
        )
        support[:, team] = raw * bystander_scale

        present = dedup_valid[:, slots] & np.isfinite(players[:, slots]).all(axis=2)
        count = present.sum(axis=1)
        masked_p = np.where(present[:, :, None], players[:, slots], 0.0)
        centre = masked_p.sum(axis=1) / np.maximum(count[:, None], 1)
        centre[count == 0] = np.nan
        centroids[:, team] = centre
        sq = np.where(present[:, :, None], (masked_p - centre[:, None, :]) ** 2, 0.0).sum(axis=(1, 2))
        spread[:, team] = np.sqrt(sq / np.maximum(count, 1))
        spread[count == 0, team] = np.nan

    # Contested duel attenuation: when both teams sit inside control zone with tiny separation,
    # instantaneous directional deltas are noisy and possession should stay with prior owner.
    both_close = usable & (nearest[:, 0] < 2.6) & (nearest[:, 1] < 2.6)
    safe_n0 = np.where(both_close, nearest[:, 0], 0.0)
    safe_n1 = np.where(both_close, nearest[:, 1], 0.0)
    closeness = np.where(both_close, np.exp(-np.maximum(safe_n0, safe_n1) / 2.2), 0.0)
    balance = np.where(both_close, np.exp(-np.abs(safe_n0 - safe_n1) / 0.70), 0.0)
    contest_factor = np.clip(closeness * balance, 0.0, 1.0)

    raw_delta = support[:, 0] - support[:, 1]
    attenuated_delta = raw_delta * (1.0 - 0.35 * contest_factor) * reliability

    # Crisp control anchors: clear margin, close control, non-contested.
    crisp_h = (
        usable
        & (nearest[:, 0] < 2.0)
        & (nearest[:, 1] > nearest[:, 0] + 0.55)
        & (contest_factor < 0.45)
        & (~np.isfinite(relative[:, 0]) | (relative[:, 0] < 5.0))
    )
    crisp_a = (
        usable
        & (nearest[:, 1] < 2.0)
        & (nearest[:, 0] > nearest[:, 1] + 0.55)
        & (contest_factor < 0.45)
        & (~np.isfinite(relative[:, 1]) | (relative[:, 1] < 5.0))
    )

    # Same-team pass flight continuity bridge across gaps/flights <= 2.2 s.
    anchor = np.full(n, -1, dtype=np.int8)
    anchor[crisp_h] = 0
    anchor[crisp_a] = 1
    pass_bridge = np.zeros(n, dtype=float)
    max_pass_gap = max(1, round(2.2 * fps))
    for g_a, g_b in runs(anchor == -1):
        if g_a == 0 or g_b == n or (g_b - g_a) > max_pass_gap:
            continue
        left_team = int(anchor[g_a - 1])
        right_team = int(anchor[g_b])
        if left_team == right_team and left_team in (0, 1):
            # Stronger bridge when ball is fast or occluded between same-team crisp controls.
            sign = 1.0 if left_team == 0 else -1.0
            pass_bridge[g_a:g_b] = sign * 0.45

    # Control acquisition detector for exact transition snapping and deflection rejection.
    # A genuine acquisition requires sustained forward superiority (>= 0.36 s) + trap or kick.
    horizon = max(4, round(0.44 * fps))
    acquisition_edge = np.zeros(n, dtype=float)
    for team in (0, 1):
        other = 1 - team
        lead = usable & (nearest[:, team] < 2.3) & (nearest[:, other] > nearest[:, team] + 0.45)
        lead_ratio = np.zeros(n, dtype=float)
        cs_lead = np.r_[0.0, np.cumsum(lead.astype(float))]
        cs_use = np.r_[0.0, np.cumsum(usable.astype(float))]
        idx = np.arange(n)
        hi = np.minimum(n, idx + horizon)
        use_cnt = cs_use[hi] - cs_use[idx]
        lead_ratio = np.where(use_cnt >= 2, (cs_lead[hi] - cs_lead[idx]) / np.maximum(use_cnt, 1.0), 0.0)

        # Trap or controlled pass/turn inside local neighborhood.
        low_rel = usable & np.isfinite(relative[:, team]) & (relative[:, team] < 4.5)
        cs_rel = np.r_[0.0, np.cumsum(low_rel.astype(float))]
        has_trap = (cs_rel[hi] - cs_rel[idx]) >= 2
        kick_like = (turn_sharpness > 0.35) | (
            np.isfinite(features["acceleration"]) & (features["acceleration"] > 15.0)
        )
        cs_kick = np.r_[0.0, np.cumsum(kick_like.astype(float))]
        has_kick = (cs_kick[np.minimum(n, idx + 4)] - cs_kick[np.maximum(0, idx - 2)]) >= 1

        sustained_gain = (lead_ratio >= 0.60) & (has_trap | has_kick) & lead
        # Mark onset of sustained gain (not every interior frame of an existing dribble).
        onset = sustained_gain.copy()
        if n > 1:
            prev_win = sustained(lead, max(2, round(0.24 * fps)))
            onset[1:] &= ~prev_win[:-1]
        # Allow a narrow 3-frame snapping band around onset.
        snapped = np.zeros(n, dtype=float)
        for pos in np.flatnonzero(onset):
            lo, p_hi = max(0, pos - 1), min(n, pos + 3)
            snapped[lo:p_hi] = np.maximum(snapped[lo:p_hi], np.where(has_trap[pos], 1.0, 0.75))
        acquisition_edge = np.maximum(acquisition_edge, snapped)

    return {
        "nearest": nearest,
        "density": density,
        "support": support,
        "relative_motion": relative,
        "controller_consistency": consistency,
        "centroids": centroids,
        "spread": spread,
        "contest_factor": contest_factor,
        "delta": attenuated_delta,
        "pass_bridge": pass_bridge,
        "acquisition_edge": np.clip(acquisition_edge, 0.0, 1.0),
    }


def bidirectional_uncertainty_memory(delta, reliability, contest_factor, fps, seconds):
    """Bridge loose, contested, and occluded intervals without lagging behind crisp control."""
    n = len(delta)
    if n == 0:
        return np.zeros(0, dtype=float)
    decay = float(np.exp(-1.0 / max(1.0, fps * seconds)))
    # Update faster on confident control, freeze/decay across occlusions and contests.
    rate = np.clip(0.35 * reliability * (1.0 - 0.7 * contest_factor), 0.0, 0.45)

    fwd = np.zeros(n, dtype=float)
    mem = 0.0
    for i in range(n):
        mem *= decay
        r = float(rate[i])
        if r > 0.05 and abs(delta[i]) > 0.05:
            mem = (1.0 - r) * mem + r * float(delta[i])
        fwd[i] = mem

    bwd = np.zeros(n, dtype=float)
    mem = 0.0
    for i in range(n - 1, -1, -1):
        mem *= decay
        r = float(rate[i])
        if r > 0.05 and abs(delta[i]) > 0.05:
            mem = (1.0 - r) * mem + r * float(delta[i])
        bwd[i] = mem

    # Suppress memory contribution when instantaneous control is crisp so turnovers don't lag.
    certainty = np.clip(reliability * (1.0 - 0.7 * contest_factor) * np.minimum(np.abs(delta) * 2.2, 1.0), 0.0, 1.0)
    return (0.72 * fwd + 0.28 * bwd) * (1.0 - 0.85 * certainty)
