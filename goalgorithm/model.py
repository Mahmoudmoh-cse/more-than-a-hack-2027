"""Candidate inference pipeline preserving v1 configurations and adding modular v2 ablations."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from starter.baseline import smooth
from .control import compute_control_evidence
from .features import extract
from .quality import extract_quality_features
from .segment_decoder import decode_segments
from .stoppage import compute_stoppage_evidence, parse_tactical_context
from .tagging import dead_evidence, evidence
from .temporal import controller_memory, viterbi


def _infer_v1_half(players, ball, teams, team_names, events, first, last, half, fps, pitch, config):
    f = extract(players, ball, teams, fps, pitch, config)
    prior, tag_confidence, restart = evidence(events, first, last + 1, half, team_names, fps, config)
    delta = f["support"][:, 0] - f["support"][:, 1]
    geometry = delta * f["reliability"]
    memory = controller_memory(geometry, f["reliability"], fps, config.memory_seconds)
    team_score = config.geometry_weight * geometry + 0.25 * memory
    if config.model in {"tagging", "temporal"}:
        team_score += config.tag_weight * prior * (1.0 - 0.5 * f["reliability"])

    if config.model == "baseline":
        d = np.linalg.norm(players - ball[:, None], axis=2)
        d[~np.isfinite(d)] = np.inf
        nearest = np.argmin(d, axis=1)
        codes = np.where(np.isfinite(d.min(axis=1)), teams[nearest], -1)
        result = smooth(codes, config.baseline_window)
        dead = None
    elif config.model == "temporal":
        dead = dead_evidence(f, restart, fps, config)
        live_penalty = np.maximum(dead, 0.0)
        emissions = np.column_stack([team_score - live_penalty, -team_score - live_penalty, dead])
        states = viterbi(
            emissions,
            config.switch_cost,
            config.dead_switch_cost,
            np.minimum(np.abs(geometry), 1.0),
        )
        result = np.array(["H", "A", "D"])[states]
    else:
        dead = None
        result = np.where(team_score >= 0, "H", "A")

    if config.model != "baseline" and not np.any(np.abs(team_score) > 1e-6):
        if config.model != "temporal" or not np.any(dead > 0):
            result[:] = "X"

    diag = {
        "half": half,
        "missing_ball_fraction": float((~f["observed"]).mean()),
        "low_reliability_fraction": float((f["reliability"] < 0.2).mean()),
        "tag_coverage_fraction": float((tag_confidence > 0).mean()),
        "max_missing_ball_seconds": float(f["missing_length"].max() / fps),
        "slot_discontinuities_or_gaps": f["slot_jump_count"],
    }
    return result, diag


def _infer_v2_half(players, ball, teams, team_names, events, first, last, half, fps, pitch, config):
    use_v2_control = config.model in {"v2_control", "v2"}
    use_v2_stoppage = config.model in {"v2_stoppage", "v2"}
    use_v2_decoder = config.model in {"v2_decoder", "v2"}

    # 1. Quality / kinematics features
    if use_v2_control or use_v2_stoppage:
        qf = extract_quality_features(players, ball, teams, fps, pitch, config)
    else:
        qf = None
    v1_f = None if (use_v2_control and use_v2_stoppage) else extract(players, ball, teams, fps, pitch, config)

    # 2. Tactical Smart Tagging fusion
    if use_v2_stoppage:
        tactical = parse_tactical_context(events, first, last + 1, half, team_names, fps, config)
        prior = tactical["prior"]
        tag_confidence = tactical["confidence"]
        restart_mask = tactical["restart_mask"]
    else:
        prior, tag_confidence, restart_mask = evidence(
            events, first, last + 1, half, team_names, fps, config
        )
        tactical = None

    # 3. Control & directional team score
    if use_v2_control:
        ctrl = compute_control_evidence(qf, teams, fps, config)
        rel = qf["reliability"]
        memory = controller_memory(ctrl["delta"], rel, fps, config.memory_seconds)
        team_score = (
            config.geometry_weight * ctrl["delta"]
            + config.memory_weight * memory
            + config.pass_bridge_weight * ctrl["pass_bridge"] * (1.0 - 0.5 * rel)
            + config.tag_weight * prior * (1.0 - 0.5 * rel)
        )
        acq_edge = ctrl["acquisition_edge"]
        contest = ctrl["contest_factor"]
        occluded = (~qf["usable"]).astype(float)
    else:
        rel = v1_f["reliability"]
        delta = v1_f["support"][:, 0] - v1_f["support"][:, 1]
        geometry = delta * rel
        memory = controller_memory(geometry, rel, fps, config.memory_seconds)
        team_score = (
            config.geometry_weight * geometry
            + 0.25 * memory
            + config.tag_weight * prior * (1.0 - 0.5 * rel)
        )
        acq_edge = np.minimum(np.abs(geometry), 1.0)
        contest = np.zeros_like(team_score)
        occluded = (~v1_f["observed"]).astype(float)

    # 4. Stoppage & restart evidence
    if use_v2_stoppage:
        stop = compute_stoppage_evidence(qf, tactical, fps, config)
        dead = stop["dead_score"]
        stop_edge = stop["stop_edge"]
        restart_edge = stop["restart_edge"]
        if use_v2_control:
            team_score = team_score + config.restart_prior_weight * stop["restart_prior"] * (1.0 - 0.5 * rel)
    else:
        f_dead = v1_f if v1_f is not None else qf
        dead = dead_evidence(f_dead, restart_mask, fps, config)
        stop_edge = np.zeros_like(dead)
        restart_edge = np.zeros_like(dead)

    # 5. Emissions (dead-ball evidence attenuates and penalizes both live states)
    live_penalty = np.maximum(dead, 0.0)
    live_scale = np.where(dead > 0.0, 1.0 / (1.0 + 0.7 * dead), 1.0)
    scaled_team = team_score * live_scale
    emissions = np.column_stack([scaled_team - live_penalty, -scaled_team - live_penalty, dead])

    # 6. Temporal decoding
    if use_v2_decoder:
        states = decode_segments(
            emissions,
            switch_cost=config.switch_cost,
            dead_enter_cost=config.dead_enter_cost,
            dead_exit_cost=config.dead_exit_cost,
            acquisition_edge=acq_edge,
            contest_factor=contest,
            stop_edge=stop_edge,
            restart_edge=restart_edge,
            occluded=occluded,
        )
    else:
        states = viterbi(
            emissions,
            config.switch_cost,
            config.dead_switch_cost,
            np.minimum(np.abs(acq_edge), 1.0),
        )

    result = np.array(["H", "A", "D"])[states]
    if not np.any(np.abs(team_score) > 1e-6) and not np.any(dead > 0.0):
        result[:] = "X"

    feat_diag = qf if qf is not None else v1_f
    diag = {
        "half": half,
        "missing_ball_fraction": float((~feat_diag["observed"]).mean()),
        "low_reliability_fraction": float((feat_diag["reliability"] < 0.2).mean()),
        "tag_coverage_fraction": float((tag_confidence > 0).mean()),
        "max_missing_ball_seconds": float(feat_diag["missing_length"].max() / fps),
        "slot_discontinuities_or_gaps": feat_diag["slot_jump_count"],
    }
    return result, diag


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
    v1_models = {"baseline", "geometry", "tagging", "temporal"}
    for half, first, last in blocks:
        if not (0 <= first <= last < n) or first <= previous_end or half not in (1, 2):
            raise ValueError("invalid or overlapping half boundaries")
        previous_end = last
        p_slice = tracking.players[first:last + 1]
        b_slice = tracking.ball[first:last + 1]
        if config.model in v1_models:
            result, diag = _infer_v1_half(
                p_slice, b_slice, tracking.teams, tracking.team_names,
                events, first, last, half, fps, pitch, config
            )
        else:
            result, diag = _infer_v2_half(
                p_slice, b_slice, tracking.teams, tracking.team_names,
                events, first, last, half, fps, pitch, config
            )
        labels[first:last + 1] = result
        diagnostics["halves"].append(diag)
    return labels, diagnostics
