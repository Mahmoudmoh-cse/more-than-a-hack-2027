"""Signal quality, outlier rejection, bounded interpolation, centered kinematics, and slot deduplication."""

import numpy as np

from .features import runs


def remove_outlier_spikes(ball, fps, max_speed, slack):
    """Mask isolated 1-frame and 2-frame teleport glitches without hiding post-gap returns."""
    cleaned = np.array(ball, dtype=float, copy=True)
    n = len(cleaned)
    step_max = max_speed / fps + slack
    if n > 2:
        steps = np.linalg.norm(np.diff(cleaned, axis=0), axis=1)
        jump = steps > step_max
        isolated_1 = jump[:-1] & jump[1:]
        if isolated_1.any():
            bridge_1 = np.linalg.norm(cleaned[2:] - cleaned[:-2], axis=1) <= 2.0 * step_max
            mask_1 = isolated_1 & bridge_1
            cleaned[1:-1][mask_1] = np.nan
    if n > 3:
        steps = np.linalg.norm(np.diff(cleaned, axis=0), axis=1)
        jump = steps > step_max
        isolated_2 = jump[:-2] & jump[2:]
        if isolated_2.any():
            bridge_2 = np.linalg.norm(cleaned[3:] - cleaned[:-3], axis=1) <= 3.0 * step_max
            mask_2 = isolated_2 & bridge_2
            cleaned[1:-2][mask_2] = np.nan
            cleaned[2:-1][mask_2] = np.nan
    observed = np.isfinite(cleaned).all(axis=1)
    return cleaned, observed


def interpolate_short_gaps(ball, observed, fps, max_gap_seconds, max_speed):
    """Linearly bridge short tracking dropouts while keeping long gaps explicitly missing."""
    filled = np.array(ball, dtype=float, copy=True)
    n = len(filled)
    reliability = observed.astype(float)
    missing_length = np.zeros(n, dtype=np.int32)
    max_gap = max(0, round(max_gap_seconds * fps))
    for first, end in runs(~observed):
        gap = end - first
        missing_length[first:end] = gap
        if first == 0 or end == n or gap > max_gap:
            continue
        speed = np.linalg.norm(filled[end] - filled[first - 1]) * fps / (gap + 1)
        if np.isfinite(speed) and speed <= max_speed:
            fraction = np.arange(1, gap + 1, dtype=float) / (gap + 1)
            filled[first:end] = filled[first - 1] + fraction[:, None] * (filled[end] - filled[first - 1])
            reliability[first:end] = 0.35
    usable = np.isfinite(filled).all(axis=1)
    return filled, usable, reliability, missing_length


def centered_motion(xy, fps, lag, max_speed, slack):
    """Low-latency velocity using centered differences with clean one-sided segment edges.

    Avoids the systematic backward-difference lag on restarts and boundary exits while
    invalidating derivatives across missing observations or slot/teleport jumps.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    velocity = np.full_like(xy, np.nan)
    if n < 2:
        return velocity, np.zeros(max(0, n - 1), dtype=bool)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    bad = ~np.isfinite(steps) | (steps > max_speed / fps + slack)
    cumulative = np.concatenate([np.zeros_like(bad[:1], dtype=int), np.cumsum(bad, axis=0)])

    half = max(1, lag // 2)
    span = 2 * half
    if n > span:
        delta = (xy[span:] - xy[:-span]) * (fps / span)
        valid = (cumulative[span:] - cumulative[:-span]) == 0
        valid &= np.linalg.norm(delta, axis=-1) <= max_speed
        velocity[half:n - half] = np.where(valid[..., None], delta, np.nan)

    edge_lag = max(1, min(lag, 4))
    if n > edge_lag:
        fwd_delta = (xy[edge_lag:] - xy[:-edge_lag]) * (fps / edge_lag)
        fwd_valid = (cumulative[edge_lag:] - cumulative[:-edge_lag]) == 0
        fwd_valid &= np.linalg.norm(fwd_delta, axis=-1) <= max_speed
        need_fwd = ~np.isfinite(velocity[:-edge_lag]).all(axis=-1) & fwd_valid
        velocity[:-edge_lag] = np.where(need_fwd[..., None], fwd_delta, velocity[:-edge_lag])

        bwd_delta = (xy[edge_lag:] - xy[:-edge_lag]) * (fps / edge_lag)
        bwd_valid = fwd_valid
        need_bwd = ~np.isfinite(velocity[edge_lag:]).all(axis=-1) & bwd_valid
        velocity[edge_lag:] = np.where(need_bwd[..., None], bwd_delta, velocity[edge_lag:])

    return velocity, bad


def deduplicate_team_slots(players, teams, min_sep=0.45, chunk_size=2048):
    """Keep at most one representative per closely overlapping same-team track cluster."""
    n, s, _ = players.shape
    valid = np.isfinite(players).all(axis=2)
    keep = valid.copy()
    thresh_sq = float(min_sep) ** 2
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if len(slots) < 2:
            continue
        team_xy = players[:, slots]
        team_valid = valid[:, slots]
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            chunk = team_xy[start:stop]  # C, M, 2
            cv = team_valid[start:stop]  # C, M
            diff = chunk[:, :, None, :] - chunk[:, None, :, :]
            dist_sq = np.sum(diff * diff, axis=3)
            # Duplicate if an earlier slot index on the same team sits within min_sep.
            m = len(slots)
            lower = np.tril(np.ones((m, m), dtype=bool), k=-1)
            pair_ok = cv[:, :, None] & cv[:, None, :] & lower[None, :, :]
            dup = np.any(pair_ok & (dist_sq < thresh_sq), axis=2)
            keep[start:stop, slots] &= ~dup
    return keep


def pitch_boundary_metrics(ball, ball_velocity, pitch):
    """Signed outside depth (>0 outside pitch), unsigned edge distance, and outward speed."""
    length, width = float(pitch[0]), float(pitch[1])
    bx, by = ball[:, 0], ball[:, 1]
    left_out = -bx
    right_out = bx - length
    top_out = -by
    bottom_out = by - width
    outside_depth = np.maximum.reduce([left_out, right_out, top_out, bottom_out])
    boundary_distance = np.minimum.reduce([np.abs(bx), np.abs(length - bx), np.abs(by), np.abs(width - by)])

    vx, vy = ball_velocity[:, 0], ball_velocity[:, 1]
    # Outward velocity along whichever boundary is closest.
    edge_idx = np.argmin(
        np.stack([np.abs(bx), np.abs(length - bx), np.abs(by), np.abs(width - by)], axis=1),
        axis=1,
    )
    outward_speed = np.select(
        [edge_idx == 0, edge_idx == 1, edge_idx == 2, edge_idx == 3],
        [-vx, vx, -vy, vy],
        default=np.nan,
    )
    return outside_depth, boundary_distance, outward_speed


def extract_quality_features(players, ball, teams, fps, pitch, config):
    """Extract clean, centered, slot-deduplicated physical features for one half."""
    cleaned_ball, observed = remove_outlier_spikes(ball, fps, config.max_ball_speed, config.position_slack)
    ball_filled, usable, reliability, missing_length = interpolate_short_gaps(
        cleaned_ball, observed, fps, config.interpolation_seconds, config.max_ball_speed
    )

    lag = max(2, round(config.velocity_seconds * fps))
    ball_velocity, ball_jumps = centered_motion(
        ball_filled, fps, lag, config.max_ball_speed, config.position_slack
    )
    player_velocity, player_jumps = centered_motion(
        players, fps, lag, config.max_player_speed, config.position_slack
    )
    ball_speed = np.linalg.norm(ball_velocity, axis=1)

    n = len(ball_filled)
    acceleration = np.full(n, np.nan)
    h = max(1, lag // 2)
    if n > 2 * h:
        dv = ball_velocity[2 * h:] - ball_velocity[:-2 * h]
        ok = np.isfinite(dv).all(axis=1)
        acceleration[h:n - h] = np.where(ok, np.linalg.norm(dv, axis=1) * (fps / (2 * h)), np.nan)

    # Sharp directional turns (kicks, deflections, tackles) via pre/post velocity cosine.
    turn_sharpness = np.zeros(n, dtype=float)
    w_turn = max(2, min(4, lag))
    if n > 2 * w_turn:
        v_in = (ball_filled[w_turn:-w_turn] - ball_filled[:-2 * w_turn]) * (fps / w_turn)
        v_out = (ball_filled[2 * w_turn:] - ball_filled[w_turn:-w_turn]) * (fps / w_turn)
        s_in = np.linalg.norm(v_in, axis=1)
        s_out = np.linalg.norm(v_out, axis=1)
        valid_turn = np.isfinite(s_in) & np.isfinite(s_out) & (s_in > 1.5) & (s_out > 1.5)
        valid_turn &= (s_in <= config.max_ball_speed) & (s_out <= config.max_ball_speed)
        dot = np.sum(v_in * v_out, axis=1)
        cos_a = np.where(valid_turn, np.clip(dot / np.maximum(s_in * s_out, 1e-6), -1.0, 1.0), 1.0)
        turn_sharpness[w_turn:-w_turn] = np.where(valid_turn, 0.5 * (1.0 - cos_a), 0.0)

    # Down-weight fast aerial motion and extreme acceleration spikes for proximity reliability.
    reliability *= np.where(np.isfinite(ball_speed), 1.0 / (1.0 + (ball_speed / 20.0) ** 2), 0.70)
    reliability *= np.where(np.isfinite(acceleration), 1.0 / (1.0 + (acceleration / 120.0) ** 2), 1.0)
    reliability[~usable] = 0.0

    dedup_valid = deduplicate_team_slots(players, teams)
    player_speed = np.linalg.norm(player_velocity, axis=2)
    valid_ps = np.isfinite(player_speed) & dedup_valid
    ps_clean = np.where(valid_ps, player_speed, 0.0)
    ps_count = valid_ps.sum(axis=1)
    player_mean_speed = np.where(ps_count >= 4, ps_clean.sum(axis=1) / np.maximum(ps_count, 1), np.nan)

    outside_depth, boundary_distance, outward_speed = pitch_boundary_metrics(
        ball_filled, ball_velocity, pitch
    )
    outside = usable & (outside_depth > config.outside_margin)

    return {
        "players": np.asarray(players, dtype=float),
        "ball": ball_filled,
        "observed": observed,
        "usable": usable,
        "reliability": reliability,
        "missing_length": missing_length,
        "ball_velocity": ball_velocity,
        "ball_speed": ball_speed,
        "acceleration": acceleration,
        "turn_sharpness": turn_sharpness,
        "player_velocity": player_velocity,
        "dedup_valid": dedup_valid,
        "player_mean_speed": player_mean_speed,
        "outside_depth": outside_depth,
        "outside": outside,
        "boundary_distance": boundary_distance,
        "outward_speed": outward_speed,
        "slot_jump_count": int(player_jumps.sum()),
        "ball_jump_count": int(ball_jumps.sum()),
    }
