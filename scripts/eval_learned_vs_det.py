"""Leave-One-Match-Out (LOMO) comparison of a compact pure-NumPy linear/logistic temporal
classifier against deterministic Goalgorithm v2 on external Metrica Sample Games 1-3.

Never touches BAR-ATH competition data or the public 5-minute ground truth.
"""

import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from goalgorithm.config import load_config
from goalgorithm.control import compute_control_evidence
from goalgorithm.quality import extract_quality_features
from goalgorithm.segment_decoder import decode_segments
from goalgorithm.stoppage import compute_stoppage_evidence, parse_tactical_context
from goalgorithm.temporal import controller_memory
from scripts.external_benchmark import (
    DEFAULT_EXTERNAL_DIR,
    FPS,
    PITCH,
    degrade_to_atd,
    load_sample_game_1_or_2,
    load_sample_game_3,
)

ROOT = Path(__file__).resolve().parents[1]


def extract_frame_feature_matrix(tracking, events, blocks, config):
    n = len(tracking.ball)
    feats = np.zeros((n, 14), dtype=np.float32)
    aux = {}
    for half, first, last in blocks:
        p = tracking.players[first:last + 1]
        b = tracking.ball[first:last + 1]
        qf = extract_quality_features(p, b, tracking.teams, FPS, PITCH, config)
        tac = parse_tactical_context(events, first, last + 1, half, tracking.team_names, FPS, config)
        stop = compute_stoppage_evidence(qf, tac, FPS, config)
        ctrl = compute_control_evidence(qf, tracking.teams, FPS, config)
        mem = controller_memory(ctrl["delta"], qf["reliability"], FPS, config.memory_seconds)
        rel = qf["reliability"]
        bs = np.nan_to_num(qf["ball_speed"], nan=0.0)
        ps = np.nan_to_num(qf["player_mean_speed"], nan=1.2)
        od = np.nan_to_num(qf["outside_depth"], nan=0.0)
        n0 = np.clip(np.nan_to_num(ctrl["nearest"][:, 0], nan=20.0, posinf=20.0), 0.0, 20.0)
        n1 = np.clip(np.nan_to_num(ctrl["nearest"][:, 1], nan=20.0, posinf=20.0), 0.0, 20.0)

        X = np.column_stack([
            ctrl["delta"],
            mem,
            ctrl["pass_bridge"] * (1.0 - 0.5 * rel),
            tac["prior"] * (1.0 - 0.5 * rel),
            stop["restart_prior"] * (1.0 - 0.5 * rel),
            stop["dead_score"],
            stop["live_strength"],
            rel,
            qf["usable"].astype(np.float32),
            bs / 20.0,
            ps / 3.0,
            np.clip(od, -5.0, 5.0),
            (n1 - n0) / 10.0,
            ctrl["contest_factor"],
        ]).astype(np.float32)
        feats[first:last + 1] = X
        aux[half] = (first, last, ctrl, stop, qf)
    return feats, aux


def fit_ridge_multiclass(X, y, mask, l2=10.0):
    """Closed-form L2-regularized 3-class linear emission model in pure NumPy."""
    Xm = X[mask].astype(np.float64)
    ym = y[mask]
    # Standardize features on training folds only.
    mu = Xm.mean(axis=0)
    sd = np.maximum(Xm.std(axis=0), 1e-3)
    Z = (Xm - mu) / sd
    Z = np.column_stack([Z, np.ones(len(Z), dtype=np.float64)])
    Y = np.full((len(ym), 3), -1.0, dtype=np.float64)
    for k in (0, 1, 2):
        Y[ym == k, k] = 1.0
    reg = np.eye(Z.shape[1], dtype=np.float64) * l2
    reg[-1, -1] = 0.0  # do not penalize intercept
    W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)
    return mu, sd, W


def predict_learned_emissions(X, mu, sd, W, scale=2.2):
    Z = (X.astype(np.float64) - mu) / sd
    Z = np.column_stack([Z, np.ones(len(Z), dtype=np.float64)])
    return (Z @ W) * scale


def main():
    cfg = load_config(ROOT / "configs" / "v2.json")
    raw_games = [
        ("Sample_Game_1", *load_sample_game_1_or_2(DEFAULT_EXTERNAL_DIR / "Sample_Game_1", "Sample_Game_1")),
        ("Sample_Game_2", *load_sample_game_1_or_2(DEFAULT_EXTERNAL_DIR / "Sample_Game_2", "Sample_Game_2")),
        ("Sample_Game_3", *load_sample_game_3(DEFAULT_EXTERNAL_DIR / "Sample_Game_3")),
    ]
    prepared = []
    for idx, (name, trk_raw, gt, blocks, sp_rec) in enumerate(raw_games):
        trk, ev = degrade_to_atd(trk_raw, gt, sp_rec, seed=100 + idx)
        X, aux = extract_frame_feature_matrix(trk, ev, blocks, cfg)
        y = pm.label_codes(gt)
        valid_mask = (y != pm.X)
        prepared.append((name, trk, ev, blocks, X, aux, gt, y, valid_mask))

    folds = []
    from goalgorithm.model import infer
    for test_i in range(len(prepared)):
        train_X = np.concatenate([prepared[j][4] for j in range(len(prepared)) if j != test_i], axis=0)
        train_y = np.concatenate([prepared[j][7] for j in range(len(prepared)) if j != test_i], axis=0)
        train_m = np.concatenate([prepared[j][8] for j in range(len(prepared)) if j != test_i], axis=0)
        mu, sd, W = fit_ridge_multiclass(train_X, train_y, train_m, l2=50.0)

        name, trk, ev, blocks, X, aux, gt, y, _ = prepared[test_i]
        em_all = predict_learned_emissions(X, mu, sd, W, scale=2.2)
        pred_learned = np.full(len(gt), "X", dtype="<U1")
        for half, first, last in blocks:
            _, _, ctrl, stop, qf = aux[half]
            st = decode_segments(
                em_all[first:last + 1],
                cfg.switch_cost, cfg.dead_enter_cost, cfg.dead_exit_cost,
                ctrl["acquisition_edge"], ctrl["contest_factor"],
                stop["stop_edge"], stop["restart_edge"], (~qf["usable"]).astype(float),
            )
            pred_learned[first:last + 1] = np.array(["H", "A", "D"])[st]

        pred_det, _ = infer(trk, ev, cfg, blocks=blocks, fps=FPS, pitch=PITCH)
        r_l = pm.evaluate(y, pm.label_codes(pred_learned))
        r_d = pm.evaluate(y, pm.label_codes(pred_det))
        folds.append({
            "held_out": name,
            "learned_spellF1": r_l["spell_f1_mean"],
            "learned_macroF1": r_l["macro_f1"],
            "learned_D_f1": r_l["per_class"]["D"]["f1"],
            "det_v2_spellF1": r_d["spell_f1_mean"],
            "det_v2_macroF1": r_d["macro_f1"],
            "det_v2_D_f1": r_d["per_class"]["D"]["f1"],
        })
        print(
            f"Fold {name}: "
            f"det_v2 spellF1={r_d['spell_f1_mean']:.4f} macroF1={r_d['macro_f1']:.4f} D_f1={r_d['per_class']['D']['f1']:.4f} | "
            f"learned+v2dec spellF1={r_l['spell_f1_mean']:.4f} macroF1={r_l['macro_f1']:.4f} D_f1={r_l['per_class']['D']['f1']:.4f}"
        )

    out = {
        "provenance": "metrica-sports/sample-data (Sample_Game_1, Sample_Game_2, Sample_Game_3), ATD-degraded",
        "folds": folds,
        "mean_det_v2_spellF1": float(np.mean([f["det_v2_spellF1"] for f in folds])),
        "mean_learned_spellF1": float(np.mean([f["learned_spellF1"] for f in folds])),
        "mean_det_v2_macroF1": float(np.mean([f["det_v2_macroF1"] for f in folds])),
        "mean_learned_macroF1": float(np.mean([f["learned_macroF1"] for f in folds])),
    }
    out_path = ROOT / "artifacts" / "external_benchmark" / "lomo_learned_vs_det.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
