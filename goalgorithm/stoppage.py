"""Event-driven stoppage, boundary-exit, restart-kick, and Smart Tagging fusion."""

import numpy as np

from .features import runs, sustained
from .tagging import DEFENSIVE, OFFENSIVE, team_index


SET_PIECE_TYPES = {
    "throw-in": "throw-in",
    "throw in": "throw-in",
    "corner kick": "corner",
    "corner": "corner",
    "goal kick": "goal-kick",
    "free kick": "free-kick",
    "kick off (start)": "kickoff-start",
    "kick off (after goal)": "kickoff-goal",
    "kick off": "kickoff",
}


def _normalize_half(value):
    text = str(value or "").strip().casefold()
    if text in {"", "n/a", "na", "none"}:
        return None
    if text in {"1st half", "1", "first half"}:
        return 1
    if text in {"2nd half", "2", "second half"}:
        return 2
    return -1  # unrecognized explicit half -> mismatch


def _normalize_sp_type(value):
    text = str(value or "").strip().casefold()
    for key, norm in SET_PIECE_TYPES.items():
        if key in text:
            return norm
    return "generic"


def parse_tactical_context(events, first, end, half, names, fps, config):
    """Extract normalized bounded tactical phase priors and structured restart windows."""
    n = end - first
    votes = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)
    restart_mask = np.zeros(n, dtype=bool)
    set_pieces = []
    goal_ends = []

    for event in events:
        tags = event.get("tags") or {}
        tagged_half = _normalize_half(tags.get("Half"))
        if tagged_half == -1 or (tagged_half is not None and tagged_half != half):
            continue
        try:
            a, b = int(event["start_frame"]), int(event["end_frame"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if b <= a or b <= first or a >= end:
            continue
        code = str(event.get("code") or "").strip().casefold()
        low, high = max(a, first) - first, min(b, end) - first
        owner = team_index(tags.get("Team"), names)

        if code == "set pieces":
            sp_type = _normalize_sp_type(tags.get("Type"))
            set_pieces.append({
                "start": low,
                "end": high,
                "type": sp_type,
                "team": owner,
                "raw_start": a - first,
            })
            restart_mask[max(0, low - round(config.restart_lookback_seconds * fps)):high] = True
            continue

        if code == "goals":
            goal_ends.append(high)
            if owner is not None:
                votes[low:high] += 0.8 * (1.0 if owner == 0 else -1.0)
                weight[low:high] += 0.8
            continue

        if owner is None or code not in OFFENSIVE | DEFENSIVE:
            continue
        if code in DEFENSIVE:
            owner = 1 - owner
        strength = 0.65 if code in DEFENSIVE else 1.0
        votes[low:high] += strength * (1.0 if owner == 0 else -1.0)
        weight[low:high] += strength

    prior = votes / np.maximum(weight, 1.0)
    confidence = np.minimum(weight, 1.0) * np.abs(prior)
    set_pieces.sort(key=lambda sp: sp["start"])
    return {
        "prior": prior,
        "confidence": confidence,
        "restart_mask": restart_mask,
        "set_pieces": set_pieces,
        "goal_ends": sorted(goal_ends),
    }


def _bridge_short_false(mask, max_bridge):
    """Bridge brief single-frame tracking dropouts inside an otherwise continuous run."""
    out = mask.copy()
    if max_bridge <= 0 or len(out) == 0:
        return out
    n = len(out)
    for a, b in runs(~out):
        if a > 0 and b < n and (b - a) <= max_bridge:
            out[a:b] = True
    return out


def _player_min_dist(players, ball):
    diff = players - ball[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    d[~np.isfinite(d)] = np.inf
    return d.min(axis=1)


def _detect_restart_frame(sp, features, live_strength, fps):
    """Locate the first live-play restart instant inside or near a SET PIECES window."""
    n = len(features["usable"])
    usable = features["usable"]
    speed = features["ball_speed"]
    outside_depth = features["outside_depth"]
    p_speed = features["player_mean_speed"]
    ball = features["ball"]

    win_a = max(0, sp["start"] - round(1.5 * fps))
    win_b = min(n, sp["end"] + round(1.5 * fps))
    if win_b <= win_a:
        return min(n - 1, max(0, sp["start"]))

    # Find first reliable live kick/movement after either missing or stationary ball.
    inside = usable & (outside_depth <= 0.15)
    moving = inside & np.isfinite(speed) & (speed >= 2.2)
    moving_run = sustained(moving, max(2, round(0.12 * fps)))

    # If ball was visible and stationary at placement, also detect displacement >= 1.5 m.
    sub_usable = np.flatnonzero(usable[win_a:win_b])
    place_idx = None
    for rel_i in sub_usable:
        idx = win_a + int(rel_i)
        if not np.isfinite(speed[idx]) or speed[idx] < 1.0:
            place_idx = idx
            break

    for idx in range(win_a, win_b):
        if moving_run[idx]:
            # Only treat as restart if preceded by stoppage/occlusion or placement.
            pre_lo = max(0, idx - round(2.5 * fps))
            had_stop = (
                not usable[pre_lo:idx].all()
                or np.any(usable[pre_lo:idx] & np.isfinite(speed[pre_lo:idx]) & (speed[pre_lo:idx] < 0.9))
                or np.any(usable[pre_lo:idx] & (outside_depth[pre_lo:idx] > -0.3))
            )
            if had_stop:
                return idx
        if place_idx is not None and idx > place_idx and usable[idx]:
            if np.linalg.norm(ball[idx] - ball[place_idx]) >= 1.5 and (
                not np.isfinite(speed[idx]) or speed[idx] >= 1.5
            ):
                return idx

    # Fallback when ball disappears before/across the set piece and reappears in play.
    reappear = np.flatnonzero((~usable[max(0, win_a):win_b - 1]) & inside[max(0, win_a) + 1:win_b])
    if len(reappear):
        return win_a + int(reappear[0]) + 1

    # Purely occluded set piece: use collective player acceleration surge if visible.
    w = max(5, round(0.8 * fps))
    if win_b - win_a > 2 * w:
        best_idx = None
        best_jump = 0.35
        for idx in range(max(w, win_a), min(n - w, win_b)):
            pre_ps = p_speed[idx - w:idx]
            post_ps = p_speed[idx:idx + w]
            if np.isfinite(pre_ps).any() and np.isfinite(post_ps).any():
                jump = float(np.nanmean(post_ps) - np.nanmean(pre_ps))
                if jump > best_jump:
                    best_jump = jump
                    best_idx = idx
        if best_idx is not None:
            return best_idx

    # Otherwise anchor in the latter third of the nominal Smart Tagging window.
    return min(n - 1, max(0, int(round(0.35 * sp["start"] + 0.65 * sp["end"]))))


def compute_stoppage_evidence(features, tactical, fps, config):
    """Compute positive dead-ball evidence, positive live-play evidence, and transition edges.

    Distinguishes positive stoppage evidence, positive live-play evidence, and uncertain
    occlusions so long live occlusions do not fabricate dead-ball periods.
    """
    usable = features["usable"]
    observed = features["observed"]
    speed = features["ball_speed"]
    outside_depth = features["outside_depth"]
    outward_speed = features["outward_speed"]
    boundary_dist = features["boundary_distance"]
    p_speed = features["player_mean_speed"]
    p_min_dist = _player_min_dist(features["players"], features["ball"])
    n = len(usable)

    # 1. Positive live-play evidence: ball inside pitch & moving or controlled by active players.
    inside = usable & (outside_depth <= -0.35)
    fast_inside = inside & np.isfinite(speed) & (speed >= 2.0)
    dribble_inside = inside & np.isfinite(speed) & (speed >= 0.85) & (p_min_dist <= 2.3)
    slow_controlled = (
        usable
        & (outside_depth <= -1.2)
        & (p_min_dist <= 1.8)
        & np.isfinite(p_speed)
        & (p_speed >= 1.25)
    )
    live_strength = np.zeros(n, dtype=float)
    live_strength[slow_controlled] = 0.55
    live_strength[dribble_inside] = np.maximum(live_strength[dribble_inside], 0.80)
    live_strength[fast_inside] = 1.0
    live_sustained = sustained(live_strength >= 0.70, max(3, round(0.36 * fps)))

    # 2. Out-of-play sustained detection (bridged over tiny 2-frame glitches).
    out_raw = usable & (outside_depth > 0.20)
    out_bridged = _bridge_short_false(out_raw, max(1, round(0.16 * fps)))
    out_sustained = sustained(out_bridged, max(3, round(0.28 * fps)))
    oob_score = np.zeros(n, dtype=float)
    oob_score[out_sustained] = 2.6

    stop_edge = np.zeros(n, dtype=float)
    restart_edge = np.zeros(n, dtype=float)
    for a, _ in runs(out_sustained):
        stop_edge[a] = max(stop_edge[a], 0.95)

    # 3. Boundary exit immediately followed by missing-ball gap (crucial for Metrica ATD).
    exit_gap_score = np.zeros(n, dtype=float)
    min_exit_gap = max(6, round(0.40 * fps))
    max_untagged_gap = round(22.0 * fps)
    decay_start = round(12.0 * fps)
    decay_tau = max(1.0, 6.0 * fps)
    sp_windows = tactical["set_pieces"]

    for g_a, g_b in runs(~usable):
        gap = g_b - g_a
        if gap < min_exit_gap or g_a == 0:
            continue
        # Inspect last few usable frames before disappearance.
        look_lo = max(0, g_a - max(4, round(0.32 * fps)))
        vis_idx = np.flatnonzero(usable[look_lo:g_a])
        if not len(vis_idx):
            continue
        u = look_lo + int(vis_idx[-1])
        depth_u = float(outside_depth[u])
        out_v = float(outward_speed[u]) if np.isfinite(outward_speed[u]) else 0.0
        # Check either already past boundary or fast outward trajectory crossing within ~0.45s.
        projected_depth = depth_u + max(0.0, out_v) * (0.45 if gap >= round(1.2 * fps) else 0.32)
        exited = (depth_u > 0.05) or (
            depth_u > (-2.2 if gap >= round(1.2 * fps) else -1.0)
            and out_v >= (2.8 if depth_u < -1.0 else 1.5)
            and projected_depth > 0.15
            and (p_min_dist[u] > 0.85 or out_v >= 3.5)
        )
        if not exited:
            continue
        # Walk back a couple frames to catch exact boundary crossing frame if visible.
        exit_start = u
        for k in range(u, max(0, u - round(0.5 * fps)) - 1, -1):
            if usable[k] and outside_depth[k] > 0.0:
                exit_start = k
            elif outside_depth[k] <= 0.0:
                break
        # Do not extend past a linked set-piece restart if one overlaps this gap.
        cap_end = min(g_b, g_a + max_untagged_gap)
        for sp in sp_windows:
            if g_a - round(2.0 * fps) <= sp["end"] and sp["start"] <= g_b + round(3.0 * fps):
                cap_end = min(g_b, sp["end"] + round(1.5 * fps))
                break
        stop_edge[exit_start] = max(stop_edge[exit_start], 1.0)
        span_len = max(1, cap_end - exit_start)
        dt = np.arange(span_len, dtype=float)
        conf = np.where(dt <= decay_start, 2.35, 2.35 * np.exp(-(dt - decay_start) / decay_tau))
        exit_gap_score[exit_start:cap_end] = np.maximum(exit_gap_score[exit_start:cap_end], conf)

    # 4. Event-driven Set-Piece stoppage preparation and restart snapping.
    sp_score = np.zeros(n, dtype=float)
    restart_prior = np.zeros(n, dtype=float)
    max_lookback = round(45.0 * fps)
    for sp in sp_windows:
        t_res = _detect_restart_frame(sp, features, live_strength, fps)
        pre_lo = max(0, t_res - round(3.0 * fps))
        if pre_lo >= t_res:
            continue
        # Require physical stoppage evidence (gap, stillness, outside, or quiet players)
        # so contradictory live play tags never invent stoppages.
        has_gap = int((~usable[pre_lo:t_res]).sum()) >= round(0.35 * fps)
        still_mask = usable & np.isfinite(speed) & (speed < 0.85)
        has_still = int(still_mask[pre_lo:t_res].sum()) >= round(0.45 * fps)
        has_out = int((usable[pre_lo:t_res] & (outside_depth[pre_lo:t_res] > -0.6)).sum()) >= 2
        quiet_players = (
            np.isfinite(p_speed[pre_lo:t_res]).any()
            and float(np.nanmedian(p_speed[pre_lo:t_res])) < 1.35
            and int((~live_sustained[pre_lo:t_res]).sum()) >= round(1.0 * fps)
        )
        if not (has_gap or has_still or has_out or quiet_players):
            continue

        # Walk backward from t_res - 1 until encountering live play.
        back_limit = max(0, t_res - max_lookback)
        stop_start = t_res
        seen_gap = False
        for k in range(t_res - 1, back_limit - 1, -1):
            if not usable[k]:
                seen_gap = True
            if live_sustained[k]:
                break
            # Re-entering an interior controlled/visible ball before a missing-ball gap
            # means the gap itself was the out-of-play stretch; stop at the gap start.
            if seen_gap and usable[k] and outside_depth[k] <= -0.8 and p_min_dist[k] <= 1.2:
                break
            # In an uninformative gap without boundary exit, stop if players are sprinting live.
            if (not usable[k]) and np.isfinite(p_speed[k]) and p_speed[k] > 2.55:
                ahead = max(back_limit, k - round(1.5 * fps))
                if np.isfinite(p_speed[ahead:k + 1]).all() and float(np.mean(p_speed[ahead:k + 1])) > 2.45:
                    break
            stop_start = k
            if stop_edge[k] >= 0.9:
                break

        if t_res > stop_start:
            sp_score[stop_start:t_res] = np.maximum(sp_score[stop_start:t_res], 2.4)
            stop_edge[stop_start] = max(stop_edge[stop_start], 0.85)
            restart_edge[t_res] = max(restart_edge[t_res], 1.0)
            if sp["team"] is not None:
                post_hi = min(n, t_res + round(2.5 * fps))
                sign = 1.0 if sp["team"] == 0 else -1.0
                dt = np.arange(post_hi - t_res, dtype=float)
                restart_prior[t_res:post_hi] += sign * np.exp(-dt / (1.5 * fps))

        # Pre-kickoff dead interval at start of a graded half if Kick Off (Start) is delayed.
        if sp["type"] == "kickoff-start" and sp["start"] > round(2.0 * fps):
            if not np.any(live_sustained[:sp["start"]]):
                sp_score[:t_res] = np.maximum(sp_score[:t_res], 2.5)
                restart_edge[t_res] = max(restart_edge[t_res], 1.0)

    # 5. Post-goal stoppage context between GOALS end and following Kick Off (After Goal).
    for g_end in tactical["goal_ends"]:
        ko = next(
            (sp for sp in sp_windows if sp["start"] >= g_end and sp["start"] <= g_end + round(65.0 * fps)
             and "kickoff" in sp["type"]),
            None,
        )
        if ko is not None:
            t_res = _detect_restart_frame(ko, features, live_strength, fps)
            g_start = min(n, g_end + round(0.5 * fps))
            if t_res > g_start:
                dead_like = (~usable[g_start:t_res]) | (
                    np.isfinite(speed[g_start:t_res]) & (speed < 1.0)
                )
                sp_score[g_start:t_res] = np.where(
                    dead_like, np.maximum(sp_score[g_start:t_res], 2.0), sp_score[g_start:t_res]
                )
                restart_edge[t_res] = max(restart_edge[t_res], 1.0)

    # 6. Untagged stationary boundary stoppages (stationary on/near touchline + quiet motion).
    stationary_70 = usable & np.isfinite(speed) & (speed < 0.70)
    stationary_70 = _bridge_short_false(stationary_70, max(1, round(0.24 * fps)))
    quiet_b = sustained(
        stationary_70
        & ((outside_depth > -0.8) | ((boundary_dist < 2.0) & (p_min_dist > 1.5)))
        & (np.isnan(p_speed) | (p_speed < 0.95)),
        max(1, round(4.0 * fps)),
    )
    quiet_score = np.zeros(n, dtype=float)
    quiet_score[quiet_b] = 1.6
    for a, _ in runs(quiet_b):
        stop_edge[a] = max(stop_edge[a], 0.75)

    # Explicit 3-way integration: positive dead vs positive live vs uncertain (-0.65).
    # Collective high player speed during occlusions provides mild symmetric live-play evidence.
    player_live = np.where(np.isfinite(p_speed), np.clip((p_speed - 1.55) / 1.4, 0.0, 1.0), 0.0)
    base_score = -0.65 - 1.15 * live_strength - 0.55 * player_live * (1.0 - live_strength)
    pos_dead = np.maximum.reduce([oob_score, exit_gap_score, sp_score, quiet_score])
    # Suppress untagged weak dead candidates if unmistakable live play is occurring.
    dead_score = np.where(pos_dead > 0.0, np.where((pos_dead < 2.0) & (live_strength >= 0.85), base_score, pos_dead), base_score)

    return {
        "dead_score": dead_score,
        "live_strength": live_strength,
        "stop_edge": np.clip(stop_edge, 0.0, 1.0),
        "restart_edge": np.clip(restart_edge, 0.0, 1.0),
        "restart_prior": np.clip(restart_prior, -1.0, 1.0),
    }
