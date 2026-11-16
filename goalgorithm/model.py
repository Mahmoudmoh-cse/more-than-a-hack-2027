"""Four fixed candidates; the primary is temporal fusion, not selected on public GT."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from starter.baseline import smooth
from .features import extract
from .tagging import evidence, dead_evidence
from .temporal import controller_memory, viterbi


def infer(tracking, events, config, blocks=None, fps=None, pitch=None):
    config.validate()
    game = GAMES[GAME]
    fps = fps or game["fps"]
    pitch = pitch or game["pitch"]
    blocks = half_video_blocks(GAME) if blocks is None else blocks
    n = len(tracking.ball)
    labels = np.full(n, "X", dtype="<U1")
    diagnostics = {"model": config.model, "halves": [], "invalid_coordinates": tracking.invalid_coordinates}
    previous_end = -1
    for half, first, last in blocks:
        if not (0 <= first <= last < n) or first <= previous_end or half not in (1, 2):
            raise ValueError("invalid or overlapping half boundaries")
        previous_end = last
        f = extract(tracking.players[first:last+1], tracking.ball[first:last+1],
                    tracking.teams, fps, pitch, config)
        prior, tag_confidence, restart = evidence(events, first, last+1, half,
                                                 tracking.team_names, fps, config)
        delta = f["support"][:, 0]-f["support"][:, 1]
        geometry = delta*f["reliability"]
        memory = controller_memory(geometry, f["reliability"], fps, config.memory_seconds)
        team_score = config.geometry_weight*geometry + 0.25*memory
        if config.model in {"tagging", "temporal"}:
            # Tags matter more when tracking is weak but remain bounded.
            team_score += config.tag_weight*prior*(1-0.5*f["reliability"])
        if config.model == "baseline":
            # Exact published fallback is implemented by CLI using official raw reader.
            # In-memory synthetic analogue uses unprocessed raw distances.
            d = np.linalg.norm(tracking.players[first:last+1]-tracking.ball[first:last+1, None], axis=2)
            d[~np.isfinite(d)] = np.inf
            nearest = np.argmin(d, axis=1)
            codes = np.where(np.isfinite(d.min(axis=1)), tracking.teams[nearest], -1)
            result = smooth(codes, config.baseline_window)
        elif config.model == "temporal":
            dead = dead_evidence(f, restart, fps, config)
            # Proximity still identifies a team standing over a dead ball. Positive
            # stoppage evidence must lower BOTH live-team scores independently.
            live_penalty = np.maximum(dead, 0.)
            emissions = np.column_stack([team_score-live_penalty, -team_score-live_penalty, dead])
            states = viterbi(emissions, config.switch_cost, config.dead_switch_cost,
                             np.minimum(abs(geometry), 1.))
            result = np.array(["H", "A", "D"])[states]
        else:
            result = np.where(team_score >= 0, "H", "A")
        # If a whole half has no directional or dead-ball evidence, abstain rather than
        # allowing deterministic tie-breaking to invent home possession.
        if config.model != "baseline" and not np.any(abs(team_score) > 1e-6):
            if config.model != "temporal" or not np.any(dead > 0):
                result[:] = "X"
        labels[first:last+1] = result
        diagnostics["halves"].append({"half": half, "missing_ball_fraction": float((~f["observed"]).mean()),
            "low_reliability_fraction": float((f["reliability"] < 0.2).mean()),
            "tag_coverage_fraction": float((tag_confidence > 0).mean()),
            "max_missing_ball_seconds": float(f["missing_length"].max()/fps),
            "slot_discontinuities_or_gaps": f["slot_jump_count"]})
    return labels, diagnostics
