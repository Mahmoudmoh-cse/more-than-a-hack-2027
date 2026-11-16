"""Metric geometry and reliability features, computed independently in each half."""

import numpy as np


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def sustained(mask, minimum):
    out = np.zeros(len(mask), dtype=bool)
    for first, end in runs(mask):
        if end - first >= minimum:
            out[first:end] = True
    return out


def rolling_mean(values, radius):
    values = np.asarray(values, dtype=float)
    total = np.r_[0., np.cumsum(values)]
    indices = np.arange(len(values))
    low, high = np.maximum(0, indices-radius), np.minimum(len(values), indices+radius+1)
    return (total[high] - total[low]) / (high-low)


def motion(xy, fps, lag, max_speed, slack):
    """Velocity over a short window, invalidated by any intervening slot jump/gap."""
    velocity = np.full_like(xy, np.nan)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    bad = ~np.isfinite(steps) | (steps > max_speed/fps + slack)
    cumulative = np.concatenate([np.zeros_like(bad[:1], dtype=int), np.cumsum(bad, axis=0)])
    if len(xy) > lag:
        delta = (xy[lag:] - xy[:-lag]) * fps/lag
        valid = (cumulative[lag:] - cumulative[:-lag] == 0)
        valid &= np.linalg.norm(delta, axis=-1) <= max_speed
        velocity[lag:] = np.where(valid[..., None], delta, np.nan)
    return velocity, bad


def extract(players, ball, teams, fps, pitch, config):
    ball = ball.copy()
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    reliability = observed.astype(float)
    # Reject isolated teleports without removing a valid trajectory after a tracking gap.
    if n > 2:
        steps = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        jump = steps > config.max_ball_speed/fps + config.position_slack
        isolated = jump[:-1] & jump[1:]
        ball[1:-1][isolated] = np.nan
        reliability[1:-1][isolated] = 0
    accepted = np.isfinite(ball).all(axis=1)
    missing_length = np.zeros(n, dtype=np.int32)
    for first, end in runs(~accepted):
        missing_length[first:end] = end-first
        gap = end-first
        if first == 0 or end == n or gap > round(config.interpolation_seconds*fps):
            continue
        speed = np.linalg.norm(ball[end]-ball[first-1])*fps/(gap+1)
        if speed <= config.max_ball_speed:
            fraction = np.arange(1, gap+1)/(gap+1)
            ball[first:end] = ball[first-1] + fraction[:, None]*(ball[end]-ball[first-1])
            reliability[first:end] = 0.35
    lag = max(1, round(config.velocity_seconds*fps))
    ball_velocity, ball_jumps = motion(ball, fps, lag, config.max_ball_speed, config.position_slack)
    player_velocity, player_jumps = motion(players, fps, lag, config.max_player_speed, config.position_slack)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    acceleration = np.full(n, np.nan)
    if n > lag:
        acceleration[lag:] = np.linalg.norm(ball_velocity[lag:]-ball_velocity[:-lag], axis=1)*fps/lag
    # Fast projected aerial motion and implausible acceleration are weak control evidence.
    reliability *= np.where(np.isfinite(ball_speed), 1/(1+(ball_speed/18.)**2), 0.65)
    reliability *= np.where(np.isfinite(acceleration), 1/(1+(acceleration/120.)**2), 1.)
    distances = np.linalg.norm(players-ball[:, None], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    nearest = np.full((n, 2), np.inf)
    density = np.zeros((n, 2))
    support = np.zeros((n, 2))
    relative = np.full((n, 2), np.nan)
    consistency = np.zeros((n, 2))
    centroids = np.full((n, 2, 2), np.nan)
    spread = np.full((n, 2), np.nan)
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        ds = distances[:, slots]
        selected = slots[np.argmin(ds, axis=1)]
        rows = np.arange(n)
        nearest[:, team] = distances[rows, selected]
        density[:, team] = (ds < config.density_radius).sum(axis=1)
        relative[:, team] = np.linalg.norm(player_velocity[rows, selected]-ball_velocity, axis=1)
        controller = players[rows, selected]
        consistent = np.zeros(n, dtype=bool)
        if n > 1:
            consistent[1:] = np.linalg.norm(np.diff(controller, axis=0), axis=1) < config.max_player_speed/fps + config.position_slack
        consistency[:, team] = rolling_mean(consistent, lag)
        co_motion = np.where(np.isfinite(relative[:, team]), np.exp(-relative[:, team]/6.), 0.3)
        support[:, team] = np.exp(-nearest[:, team]/config.control_radius) * (
            0.65 + 0.2*co_motion + 0.15*consistency[:, team])
        present = np.isfinite(players[:, slots]).all(axis=2)
        count = present.sum(axis=1)
        centre = np.nansum(players[:, slots], axis=1)/np.maximum(count[:, None], 1)
        centre[count == 0] = np.nan
        centroids[:, team] = centre
        sq = np.nansum((players[:, slots]-centre[:, None])**2, axis=(1, 2))
        spread[:, team] = np.sqrt(sq/np.maximum(count, 1))
        spread[count == 0, team] = np.nan
    player_speed = np.linalg.norm(player_velocity, axis=2)
    valid_speed = np.isfinite(player_speed)
    mean_speed = np.nansum(player_speed, axis=1)/np.maximum(valid_speed.sum(axis=1), 1)
    mean_speed[valid_speed.sum(axis=1) < 4] = np.nan
    length, width = pitch
    outside = (ball[:, 0] < -config.outside_margin) | (ball[:, 0] > length+config.outside_margin)
    outside |= (ball[:, 1] < -config.outside_margin) | (ball[:, 1] > width+config.outside_margin)
    boundary_distance = np.minimum.reduce([abs(ball[:, 0]), abs(length-ball[:, 0]), abs(ball[:, 1]), abs(width-ball[:, 1])])
    return dict(ball=ball, observed=observed, reliability=reliability,
                missing_length=missing_length, ball_speed=ball_speed, acceleration=acceleration,
                nearest=nearest, density=density, support=support, relative_motion=relative,
                controller_consistency=consistency, centroids=centroids, spread=spread,
                player_mean_speed=mean_speed, outside=outside, boundary_distance=boundary_distance,
                slot_jump_count=int(player_jumps.sum()), ball_jump_count=int(ball_jumps.sum()))
