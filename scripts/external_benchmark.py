"""Independent external evaluation on public Metrica Sample Games 1-3 (never touches BAR-ATH).

Evaluates both raw open tracking and realistic ATD-degraded + noisy-tagging simulations
using the unchanged official evaluation/possession_metrics.py scorer.
"""

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from goalgorithm.config import Config, load_config
from goalgorithm.data import Tracking
from goalgorithm.model import infer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DIR = ROOT / "artifacts" / "external_data" / "metrica"
PITCH = (105.0, 68.0)
FPS = 25


def _sub_names(event):
    st = event.get("subtypes")
    if isinstance(st, dict):
        return {st.get("name")}
    if isinstance(st, list):
        return {s.get("name") for s in st}
    return set()


def load_sample_game_1_or_2(game_dir, game_name):
    ev = pd.read_csv(game_dir / f"{game_name}_RawEventsData.csv")
    ht = pd.read_csv(game_dir / f"{game_name}_RawTrackingData_Home_Team.csv", skiprows=2)
    at = pd.read_csv(game_dir / f"{game_name}_RawTrackingData_Away_Team.csv", skiprows=2)
    n = len(ht)

    h_cols = [c for c in ht.columns if c.startswith("Player")]
    a_cols = [c for c in at.columns if c.startswith("Player")]
    h_idx = [ht.columns.get_loc(c) for c in h_cols]
    a_idx = [at.columns.get_loc(c) for c in a_cols]

    players = np.full((n, len(h_idx) + len(a_idx), 2), np.nan, dtype=np.float32)
    for k, ci in enumerate(h_idx):
        players[:, k, 0] = ht.iloc[:, ci].to_numpy(dtype=np.float32)
        players[:, k, 1] = ht.iloc[:, ci + 1].to_numpy(dtype=np.float32)
    off = len(h_idx)
    for k, ci in enumerate(a_idx):
        players[:, off + k, 0] = at.iloc[:, ci].to_numpy(dtype=np.float32)
        players[:, off + k, 1] = at.iloc[:, ci + 1].to_numpy(dtype=np.float32)
    ball = np.column_stack([
        ht["Ball"].to_numpy(dtype=np.float32),
        ht.iloc[:, ht.columns.get_loc("Ball") + 1].to_numpy(dtype=np.float32),
    ])

    players *= np.asarray(PITCH, dtype=np.float32)
    ball *= np.asarray(PITCH, dtype=np.float32)
    teams = np.array([0] * len(h_idx) + [1] * len(a_idx), dtype=np.int8)
    tracking = Tracking(players, ball, teams, ("Home", "Away"), 0)

    p_bounds = {
        p: (int(np.flatnonzero(ht["Period"].to_numpy() == p)[0]),
            int(np.flatnonzero(ht["Period"].to_numpy() == p)[-1]))
        for p in (1, 2)
    }

    dead = np.zeros(n, dtype=bool)
    sp_records = []
    for i, row in ev.iterrows():
        t = str(row["Type"])
        st = "" if pd.isna(row["Subtype"]) else str(row["Subtype"])
        p = int(row["Period"])
        if t == "SET PIECE" or "GOAL KICK" in st or "KICK OFF" in st:
            sf = max(0, int(row["Start Frame"]) - 1)
            sp_type = (
                "Throw-in" if "THROW IN" in st else
                "Corner Kick" if "CORNER KICK" in st else
                "Goal Kick" if "GOAL KICK" in st else
                "Kick Off (Start)" if "KICK OFF" in st else
                "Free Kick"
            )
            sp_team = "N/A" if sp_type == "Throw-in" else ("Home" if row["Team"] == "Home" else "Away")
            sp_records.append((p, sf, sp_type, sp_team))
        if t in ("BALL OUT", "FAULT RECEIVED") or "OUT" in st or "OFFSIDE" in st or "HAND BALL" in st or (t == "SHOT" and "GOAL" in st) or "END HALF" in st:
            sf = max(0, int(row["End Frame"]) - 1)
            for j in range(i + 1, len(ev)):
                rj = ev.iloc[j]
                if int(rj["Period"]) != p:
                    break
                tj = str(rj["Type"])
                stj = "" if pd.isna(rj["Subtype"]) else str(rj["Subtype"])
                if tj == "SET PIECE" or "GOAL KICK" in stj or "KICK OFF" in stj:
                    rf = max(0, int(rj["Start Frame"]) - 1)
                    if rf > sf:
                        dead[sf:rf] = True
                    break
                # Live action before a set piece means advantage played / no stoppage occurred.
                if tj in ("PASS", "RECOVERY", "SHOT") and int(rj["Start Frame"]) - 1 > sf + 5:
                    break

    for p, (lo, hi) in p_bounds.items():
        pe = ev[ev["Period"] == p]
        if len(pe):
            f0 = max(lo, int(pe.iloc[0]["Start Frame"]) - 1)
            if f0 > lo:
                dead[lo:f0] = True

    def pr(r):
        t = str(r["Type"])
        st = "" if pd.isna(r["Subtype"]) else str(r["Subtype"])
        if t in ("BALL OUT", "FAULT RECEIVED"):
            return 0
        if t == "BALL LOST":
            return 1
        if t == "SET PIECE" or "GOAL KICK" in st or "KICK OFF" in st:
            return 2
        if t == "RECOVERY":
            return 3
        return 4

    live_owner = np.full(n, -1, dtype=np.int8)
    for p, (lo, hi) in sorted(p_bounds.items()):
        pe = ev[(ev["Period"] == p) & (~ev["Type"].isin(["CHALLENGE", "CARD"]))].copy()
        pe["_sf"] = pe["Start Frame"].astype(int)
        pe["_pr"] = pe.apply(pr, axis=1)
        pe = pe.sort_values(["_sf", "_pr"])
        cur = 0 if pe.iloc[0]["Team"] == "Home" else 1
        last = lo
        for _, row in pe.iterrows():
            team = 0 if row["Team"] == "Home" else 1
            sf = max(lo, int(row["Start Frame"]) - 1)
            if sf > last:
                live_owner[last:sf] = cur
            cur = team
            last = max(last, sf)
        if last <= hi:
            live_owner[last:hi + 1] = cur

    gt = np.full(n, "X", dtype="<U1")
    for p, (lo, hi) in p_bounds.items():
        gt[lo:hi + 1] = np.where(dead[lo:hi + 1], "D", np.where(live_owner[lo:hi + 1] == 0, "H", "A"))
    blocks = [(p, p_bounds[p][0], p_bounds[p][1]) for p in sorted(p_bounds)]
    return tracking, gt, blocks, sp_records


def load_sample_game_3(game_dir):
    meta = ET.parse(game_dir / "Sample_Game_3_metadata.xml").getroot()
    ids = [t.get("id") for t in meta.iter("Team")]
    p_team = {p.get("id"): ids.index(p.get("teamId")) for p in meta.iter("Player")}
    chan_to_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    first_spec = next(meta.iter("DataFormatSpecification"))
    slot_teams = [
        p_team[chan_to_player[ref.get("playerChannelId")]]
        for ref in first_spec.iter("PlayerChannelRef")
        if ref.get("playerChannelId", "").endswith("_x")
    ]
    teams = np.array(slot_teams, dtype=np.int8)
    lines = (game_dir / "Sample_Game_3_tracking.txt").read_text().splitlines()
    n = len(lines)
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    for idx, line in enumerate(lines):
        parts = line.strip().split(":")
        groups = parts[1].split(";")
        for s in range(min(len(teams), len(groups))):
            x, y = groups[s].split(",")[:2]
            players[idx, s] = (float(x), float(y))
        bx, by = parts[2].split(",")[:2]
        ball[idx] = (float(bx), float(by))
    players *= np.asarray(PITCH, dtype=np.float32)
    ball *= np.asarray(PITCH, dtype=np.float32)
    tracking = Tracking(players, ball, teams, ("Home", "Away"), 0)

    ev3 = json.loads((game_dir / "Sample_Game_3_events.json").read_text())["data"]
    p_bounds = {}
    for e in ev3:
        p = e["period"]
        s, f = e["start"]["frame"] - 1, e["end"]["frame"] - 1
        p_bounds.setdefault(p, [s, f])
        p_bounds[p][0] = min(p_bounds[p][0], s)
        p_bounds[p][1] = max(p_bounds[p][1], f)

    dead = np.zeros(n, dtype=bool)
    sp_records = []
    for i, e in enumerate(ev3):
        t = e["type"]["name"]
        st = _sub_names(e)
        p = e["period"]
        if t == "SET PIECE" or "GOAL KICK" in st or "KICK OFF" in st:
            sf = max(0, e["start"]["frame"] - 1)
            sp_type = (
                "Throw-in" if "THROW IN" in st else
                "Corner Kick" if "CORNER KICK" in st else
                "Goal Kick" if "GOAL KICK" in st else
                "Kick Off (Start)" if "KICK OFF" in st else
                "Free Kick"
            )
            sp_team = "N/A" if sp_type == "Throw-in" else ("Home" if e["team"]["id"] == "FIFATMA" else "Away")
            sp_records.append((p, sf, sp_type, sp_team))
        if t in ("BALL OUT", "FAULT RECEIVED") or "OFFSIDE" in st or "HAND BALL" in st or (t == "SHOT" and ("OUT" in st or "GOAL" in st)) or "END HALF" in st:
            sf = max(0, e["end"]["frame"] - 1)
            for j in range(i + 1, len(ev3)):
                ej = ev3[j]
                if ej["period"] != p:
                    break
                stj = _sub_names(ej)
                if ej["type"]["name"] == "SET PIECE" or "GOAL KICK" in stj or "KICK OFF" in stj:
                    rf = max(0, ej["start"]["frame"] - 1)
                    if rf > sf:
                        dead[sf:rf] = True
                    break
                if ej["type"]["name"] in ("PASS", "CARRY", "RECOVERY", "SHOT") and ej["start"]["frame"] - 1 > sf + 5:
                    break

    def pr3(e):
        t = e["type"]["name"]
        st = _sub_names(e)
        if t in ("BALL OUT", "FAULT RECEIVED"):
            return 0
        if t == "BALL LOST":
            return 1
        if t == "SET PIECE" or "GOAL KICK" in st or "KICK OFF" in st:
            return 2
        if t == "RECOVERY":
            return 3
        return 4

    live_owner = np.full(n, -1, dtype=np.int8)
    for p, (lo, hi) in sorted(p_bounds.items()):
        pe = sorted(
            [e for e in ev3 if e["period"] == p and e["type"]["name"] not in ("CHALLENGE", "CARD")],
            key=lambda e: (e["start"]["frame"], pr3(e)),
        )
        cur = 0 if pe[0]["team"]["id"] == "FIFATMA" else 1
        last = lo
        for e in pe:
            team = 0 if e["team"]["id"] == "FIFATMA" else 1
            sf = max(lo, e["start"]["frame"] - 1)
            if sf > last:
                live_owner[last:sf] = cur
            cur = team
            last = max(last, sf)
        if last <= hi:
            live_owner[last:hi + 1] = cur

    gt = np.full(n, "X", dtype="<U1")
    for p, (lo, hi) in p_bounds.items():
        gt[lo:hi + 1] = np.where(dead[lo:hi + 1], "D", np.where(live_owner[lo:hi + 1] == 0, "H", "A"))
    blocks = [(p, p_bounds[p][0], p_bounds[p][1]) for p in sorted(p_bounds)]
    return tracking, gt, blocks, sp_records


def degrade_to_atd(tracking, gt, sp_records, seed):
    """Simulate uncorrected ATD challenges: ~41% ball missingness (live aerial/box bursts + dead),
    slot permutations, duplicate slots, 15cm jitter, and noisy Smart Tagging."""
    rng = np.random.default_rng(seed)
    n, s, _ = tracking.players.shape
    players = tracking.players.copy()
    ball = tracking.ball.copy()
    teams = tracking.teams.copy()

    # Add realistic 15 cm player tracking jitter and 10 cm ball jitter.
    players += rng.normal(0.0, 0.15, size=players.shape).astype(np.float32)
    ball += rng.normal(0.0, 0.10, size=ball.shape).astype(np.float32)

    # Inject 2% duplicate ghost tracks and 3% random player dropouts, plus slot swaps every ~150 frames.
    for team in (0, 1):
        slots = np.flatnonzero(teams == team)
        if len(slots) >= 2:
            for t0 in range(0, n, 150):
                t1 = min(n, t0 + 150)
                perm = rng.permutation(slots)
                players[t0:t1, slots] = players[t0:t1][:, perm]

    drop_p = rng.random((n, s)) < 0.03
    players[drop_p] = np.nan

    # Add 2 extra duplicate track slots so tracker duplicates are exercised.
    extra_teams = np.array([0, 1], dtype=np.int8)
    extra_players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    for k, t in enumerate((0, 1)):
        src = np.flatnonzero(teams == t)[0]
        dup_mask = rng.random(n) < 0.12
        extra_players[dup_mask, k] = players[dup_mask, src] + rng.normal(0.0, 0.12, size=(int(dup_mask.sum()), 2)).astype(np.float32)
    players = np.concatenate([players, extra_players], axis=1)
    teams = np.concatenate([teams, extra_teams])

    # Biased live ball occlusions (high speed, crowded box, random bursts) reaching ~28% live missingness
    # (~43% overall missingness when combined with dead-ball gaps).
    vis = np.isfinite(ball).all(axis=1)
    bv = np.zeros(n, dtype=float)
    if n > 5:
        d = np.linalg.norm((ball[5:] - ball[:-5]) * 5.0, axis=1)
        bv[5:] = np.where(np.isfinite(d), d, 0.0)
    live = (gt == "H") | (gt == "A")
    i = 0
    while i < n:
        if live[i] and vis[i]:
            p_start = 0.018 if bv[i] > 12.0 else 0.008
            if rng.random() < p_start:
                burst = int(rng.integers(6, 55))
                ball[i:min(n, i + burst)] = np.nan
                i += burst
                continue
        i += 1

    # Inject 0.2% isolated 1-frame ball teleport glitches.
    glitch_idx = rng.choice(n, size=max(1, n // 500), replace=False)
    for gi in glitch_idx:
        if np.isfinite(ball[gi]).all():
            ball[gi] += rng.uniform(8.0, 18.0, size=2).astype(np.float32)

    # Also on 20% of dead spans where ball was missing, place a stationary visible ball near spot
    # so models cannot cheat by equating missing_ball == D!
    for p, sf, sp_type, _ in sp_records:
        if rng.random() < 0.25 and sf < n and np.isfinite(ball[sf]).all():
            span = int(rng.integers(45, 140))
            lo = max(0, sf - span)
            if np.all(gt[lo:sf] == "D"):
                ball[lo:sf] = ball[sf] + rng.normal(0.0, 0.06, size=(sf - lo, 2)).astype(np.float32)

    # Build noisy Smart Tagging events: 85% recall on set pieces with +-1.0s jitter,
    # plus noisy offensive/defensive phase tags (~10% team errors + false positives).
    events = []
    for p, sf, sp_type, sp_team in sp_records:
        if rng.random() < 0.12:
            continue  # missed set piece
        jit_a = int(rng.integers(-35, -10))
        jit_b = int(rng.integers(50, 120))
        events.append({
            "code": "SET PIECES",
            "start_frame": max(0, sf + jit_a),
            "end_frame": min(n, sf + jit_b),
            "tags": {"Half": "1st Half" if p == 1 else "2nd Half", "Type": sp_type, "Team": sp_team},
        })

    # Sample noisy phase tags along possessions.
    codes = pm.label_codes(gt)
    for a, b, code in pm.possession_spells(codes, [(0, n)]):
        if (b - a) < 75 or rng.random() < 0.35:
            continue
        team = "Home" if code == pm.H else "Away"
        if rng.random() < 0.10:
            team = "Away" if team == "Home" else "Home"  # noisy mis-tag
        half = "1st Half" if a < n // 2 else "2nd Half"
        tag_code = rng.choice(["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "ATTACKING TRANSITION", "LONG BALL"])
        events.append({
            "code": str(tag_code),
            "start_frame": max(0, a + int(rng.integers(-15, 35))),
            "end_frame": min(n, b + int(rng.integers(-25, 25))),
            "tags": {"Team": team, "Half": half},
        })

    return Tracking(players, ball, teams, tracking.team_names, 0), events


def run_benchmark(data_dir, include_degraded=True):
    games = []
    for g in ("Sample_Game_1", "Sample_Game_2"):
        gd = data_dir / g
        if gd.is_dir():
            games.append((g, *load_sample_game_1_or_2(gd, g)))
    g3 = data_dir / "Sample_Game_3"
    if g3.is_dir():
        games.append(("Sample_Game_3", *load_sample_game_3(g3)))
    if not games:
        raise FileNotFoundError(f"No external Metrica sample games found under {data_dir}")

    configs = [
        ("baseline_50", load_config(ROOT / "configs" / "baseline.json")),
        ("v1", load_config(ROOT / "configs" / "temporal.json")),
        ("v2_stoppage", load_config(ROOT / "configs" / "v2_stoppage.json")),
        ("v2_control", load_config(ROOT / "configs" / "v2_control.json")),
        ("v2_decoder", load_config(ROOT / "configs" / "v2_decoder.json")),
        ("v2", load_config(ROOT / "configs" / "v2.json")),
    ]

    regimes = [("clean_notags", False)]
    if include_degraded:
        regimes.append(("atd_noisy_tags", True))

    summary = {}
    for regime_name, degraded in regimes:
        print(f"\n=== EXTERNAL METRICA 3-MATCH BENCHMARK ({regime_name}) ===")
        regime_rows = []
        for g_idx, (gname, trk_raw, gt, blocks, sp_rec) in enumerate(games):
            if degraded:
                trk, events = degrade_to_atd(trk_raw, gt, sp_rec, seed=100 + g_idx)
                vis_rate = float(np.isfinite(trk.ball).all(axis=1).mean())
                print(f"  [{gname}] ATD ball_visible={100*vis_rate:.1f}% events={len(events)}")
            else:
                trk, events = trk_raw, []
            gt_codes = pm.label_codes(gt)
            for cname, cfg in configs:
                t0 = time.perf_counter()
                pred, _ = infer(trk, events, cfg, blocks=blocks, fps=FPS, pitch=PITCH)
                dt = time.perf_counter() - t0
                res = pm.evaluate(gt_codes, pm.label_codes(pred))
                print(
                    f"    {gname:13s} {cname:12s} spellF1={res['spell_f1_mean']:.4f} "
                    f"macroF1={res['macro_f1']:.4f} D_f1={res['per_class']['D']['f1']:.4f} "
                    f"spells={res['n_pred_spells']}/{res['n_gt_spells']} ({dt:.2f}s)",
                    flush=True,
                )
                row = {
                    "game": gname,
                    "model": cname,
                    "spellF1": res["spell_f1_mean"],
                    "macroF1": res["macro_f1"],
                    "alive_dead_f1": res["alive_dead_f1"],
                    "D_f1": res["per_class"]["D"]["f1"],
                    "team_both_alive": res["team_acc_both_alive"],
                    "transF1_25": res["transitions"][25]["f1"],
                    "mean_latency": res["transitions"][25]["mean_latency"],
                    "n_pred": res["n_pred_spells"],
                    "n_gt": res["n_gt_spells"],
                    "runtime_s": dt,
                }
                regime_rows.append(row)

        df = pd.DataFrame(regime_rows)
        agg = (
            df.groupby("model", sort=False)[
                ["spellF1", "macroF1", "alive_dead_f1", "D_f1", "team_both_alive", "transF1_25", "runtime_s"]
            ]
            .mean()
            .reset_index()
        )
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        summary[regime_name] = {"per_game": regime_rows, "mean": agg.to_dict(orient="records")}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "external_benchmark" / "results.json")
    args = parser.parse_args()
    summary = run_benchmark(args.external_dir, include_degraded=not args.clean_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full external benchmark results to {args.out}")


if __name__ == "__main__":
    main()
