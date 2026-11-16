"""Bounded noisy tactical priors; no copying phase intervals as possession labels."""

import numpy as np

from .features import sustained


OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition",
             "long ball", "ball in final third", "ball in the box", "players in the box"}
DEFENSIVE = {"defensive transition", "defending in attacking third",
             "defending in middle third", "defending in defensive third"}


def team_index(value, names):
    value = str(value or "").strip().casefold()
    if value in {"", "n/a", "na", "none", "unknown"}:
        return None
    aliases = [{"barcelona", "fc barcelona", names[0].casefold()},
               {"atletic club", "athletic club", "athletic bilbao", names[1].casefold()}]
    matches = [i for i, options in enumerate(aliases) if value in options]
    return matches[0] if len(matches) == 1 else None


def evidence(events, first, end, half, names, fps, config):
    n = end-first
    votes = np.zeros(n)
    weight = np.zeros(n)
    restart = np.zeros(n, dtype=bool)
    for event in events:
        tags = event.get("tags") or {}
        tagged_half = tags.get("Half")
        if tagged_half not in {None, "", "N/A", "1st Half", "2nd Half"}:
            continue
        if tagged_half in {"1st Half", "2nd Half"} and tagged_half != {1:"1st Half", 2:"2nd Half"}[half]:
            continue
        try:
            a, b = int(event["start_frame"]), int(event["end_frame"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if b <= a or b <= first or a >= end:
            continue
        code = str(event.get("code") or "").strip().casefold()
        low, high = max(a, first)-first, min(b, end)-first
        if code == "set pieces":
            restart[max(0, low-round(config.restart_lookback_seconds*fps)):high] = True
            # A set-piece phase includes a live restart; it is not a D label or ownership span.
            continue
        owner = team_index(tags.get("Team"), names)
        if owner is None or code not in OFFENSIVE | DEFENSIVE:
            continue
        if code in DEFENSIVE:
            owner = 1-owner
        strength = 0.7 if code in DEFENSIVE else 1.0
        votes[low:high] += strength*(1 if owner == 0 else -1)
        weight[low:high] += strength
    # Correlated tags cannot accumulate unlimited weight; conflicts cancel.
    prior = votes/np.maximum(weight, 1.)
    confidence = np.minimum(weight, 1.) * np.abs(prior)
    return prior, confidence, restart


def dead_evidence(features, restart, fps, config):
    """Stoppage requires positive evidence, never missing tracking by itself."""
    visible = features["observed"] & (features["reliability"] > 0)
    stationary = visible & (features["ball_speed"] < config.stationary_speed)
    still_run = sustained(stationary, max(1, round(config.stationary_seconds*fps)))
    outside = sustained(visible & features["outside"], max(1, round(config.outside_seconds*fps)))
    # Long, stationary ball near a boundary plus independently low player motion.
    quiet_boundary = sustained(stationary & (features["boundary_distance"] < 2.) &
                               (features["player_mean_speed"] < 0.8), round(4*fps))
    score = np.full(len(visible), -0.8)
    score[outside] = 2.2
    score[still_run & restart] = 1.8
    score[quiet_boundary] = np.maximum(score[quiet_boundary], 1.3)
    # Conjunctive tag and collective stillness may support D through an occluded restart.
    quiet_gap = sustained((~visible) & restart & (features["player_mean_speed"] < 0.5), round(2*fps))
    score[quiet_gap] = 0.8
    return score
