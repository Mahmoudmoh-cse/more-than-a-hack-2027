"""Comprehensive adversarial, behavioral, and metamorphic tests for Goalgorithm v2."""

from dataclasses import replace
import itertools
import unittest

import numpy as np

from goalgorithm.config import Config, load_config
from goalgorithm.data import Tracking
from goalgorithm.model import infer
from goalgorithm.quality import extract_quality_features
from goalgorithm.segment_decoder import build_transition_costs, decode_segments


PITCH = (105.3, 68.0)
FPS = 25


def _base_players(n, home_pos=(40.0, 30.0), away_pos=(65.0, 30.0)):
    players = np.zeros((n, 6, 2), dtype=np.float32)
    players[:, 0] = home_pos
    players[:, 1] = [home_pos[0] - 8.0, home_pos[1] - 6.0]
    players[:, 2] = [home_pos[0] + 10.0, home_pos[1] + 8.0]
    players[:, 3] = away_pos
    players[:, 4] = [away_pos[0] + 8.0, away_pos[1] + 6.0]
    players[:, 5] = [away_pos[0] - 10.0, away_pos[1] - 8.0]
    return players


def _make_tracking(n=500, owner=0, home_pos=(40.0, 30.0), away_pos=(65.0, 30.0)):
    players = _base_players(n, home_pos=home_pos, away_pos=away_pos)
    anchor = players[:, 0] if owner == 0 else players[:, 3]
    ball = anchor.copy() + np.array([0.3, 0.0], dtype=np.float32)
    teams = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    return Tracking(players, ball, teams)


def _predict_v2(tracking, events=(), model="v2", blocks=None):
    cfg = replace(Config(), model=model)
    blocks = [(1, 0, len(tracking.ball) - 1)] if blocks is None else blocks
    return infer(tracking, events, cfg, blocks=blocks, fps=FPS, pitch=PITCH)[0]


class V2StoppageTests(unittest.TestCase):
    def test_missing_ball_immediately_after_boundary_exit_is_dead(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(40.0, 4.0), away_pos=(65.0, 30.0))
        # Ball kicked toward touchline y=0 and exits at frame 150..156, then vanishes until 320.
        for t in range(140, 156):
            trk.ball[t] = [42.0, 3.0 - 0.24 * (t - 140)]
        trk.ball[156:320] = np.nan
        # At 320, play resumes inside pitch with away control.
        trk.ball[320:] = trk.players[320:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred[20:145] == "H"))
        self.assertTrue(np.all(pred[165:305] == "D"))
        self.assertTrue(np.all(pred[335:] == "A"))

    def test_ball_disappearance_during_live_interior_play_not_dead(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(50.0, 34.0))
        trk.ball[180:330] = np.nan
        pred = _predict_v2(trk)
        self.assertFalse(np.any(pred == "D"))
        self.assertTrue(np.all(pred == "H"))

    def test_players_moving_during_stoppage_still_detected(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(50.0, 2.0), away_pos=(70.0, 30.0))
        # Players jog/reposition at ~1.4 m/s while ball is missing for throw-in.
        for s in range(6):
            trk.players[:, s, 0] += np.arange(600, dtype=np.float32) * 0.056
        trk.ball[:] = trk.players[:, 0] + np.array([0.2, 0.0], dtype=np.float32)
        trk.ball[150:320] = np.nan
        events = [{"code": "SET PIECES", "start_frame": 295, "end_frame": 345,
                   "tags": {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}}]
        pred = _predict_v2(trk, events=events)
        self.assertTrue(np.all(pred[170:300] == "D"))
        self.assertTrue(np.all(pred[345:] == "H"))

    def test_throw_in_with_no_reliable_ball_observations(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(40.0, 1.5))
        trk.ball[180:340] = np.nan
        # Away takes throw-in at ~315 and controls after 340.
        trk.ball[340:] = trk.players[340:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        events = [{"code": "SET PIECES", "start_frame": 290, "end_frame": 345,
                   "tags": {"Team": "N/A", "Type": "Throw-in", "Half": "1st Half"}}]
        pred = _predict_v2(trk, events=events)
        self.assertTrue(np.all(pred[20:170] == "H"))
        self.assertTrue(np.all(pred[200:310] == "D"))
        self.assertTrue(np.all(pred[355:] == "A"))

    def test_corner_and_free_kick_preparation_and_execution(self):
        trk = _make_tracking(n=900, owner=0, home_pos=(104.5, 1.0), away_pos=(95.0, 34.0))
        # Ball stationary at corner spot 150..450, then struck fast toward away clearance at 450..490.
        trk.ball[150:450] = [105.0, 0.2]
        for t in range(450, 480):
            alpha = (t - 450) / 30.0
            trk.ball[t] = (1.0 - alpha) * np.array([105.0, 0.2]) + alpha * np.array([95.0, 34.0])
        trk.ball[480:] = trk.players[480:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        events = [{"code": "SET PIECES", "start_frame": 420, "end_frame": 480,
                   "tags": {"Team": "Barcelona", "Type": "Corner Kick", "Half": "1st Half"}}]
        pred = _predict_v2(trk, events=events)
        self.assertTrue(np.all(pred[175:440] == "D"))
        self.assertTrue(np.all(pred[495:] == "A"))

    def test_stationary_ball_during_normal_possession_not_dead(self):
        trk = _make_tracking(n=500, owner=0, home_pos=(50.0, 34.0))
        # Players moving around at >1.4 m/s while home stands near ball briefly (<3.5s)
        for s in (1, 2, 3, 4, 5):
            trk.players[:, s, 0] += np.sin(np.arange(500) * 0.08) * 1.5
        pred = _predict_v2(trk)
        self.assertFalse(np.any(pred == "D"))
        self.assertTrue(np.all(pred == "H"))

    def test_long_occlusion_interior_not_dead(self):
        trk = _make_tracking(n=1800, owner=1, away_pos=(55.0, 34.0))
        trk.ball[150:1600] = np.nan
        pred = _predict_v2(trk)
        self.assertFalse(np.any(pred == "D"))
        self.assertTrue(np.all(pred == "A"))

    def test_same_team_possession_split_by_dead_ball(self):
        trk = _make_tracking(n=700, owner=0, home_pos=(45.0, 2.0))
        trk.ball[200:400] = [45.0, -1.2]
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred[20:190] == "H"))
        self.assertTrue(np.all(pred[220:390] == "D"))
        self.assertTrue(np.all(pred[420:] == "H"))

    def test_contradictory_set_piece_tag_during_fast_live_play_ignored(self):
        trk = _make_tracking(n=500, owner=0, home_pos=(30.0, 34.0))
        trk.players[:, 0, 0] += np.arange(500, dtype=np.float32) * 0.10
        trk.ball[:, 0] = trk.players[:, 0, 0] + 0.3
        events = [{"code": "SET PIECES", "start_frame": 150, "end_frame": 300,
                   "tags": {"Team": "N/A", "Type": "Free Kick", "Half": "1st Half"}}]
        pred = _predict_v2(trk, events=events)
        self.assertFalse(np.any(pred == "D"))
        self.assertTrue(np.all(pred == "H"))

    def test_pre_kickoff_graded_frames_before_second_half_kickoff(self):
        trk = _make_tracking(n=700, owner=1, away_pos=(52.65, 34.0))
        # Graded half starts at 0, kickoff occurs at 250; ball missing before kickoff.
        trk.ball[:250] = np.nan
        for t in range(250, 700):
            trk.players[t, 3, 0] += (t - 250) * 0.09
            trk.ball[t] = trk.players[t, 3] + np.array([0.3, 0.0], dtype=np.float32)
        events = [{"code": "SET PIECES", "start_frame": 230, "end_frame": 280,
                   "tags": {"Team": "Atletic Club", "Type": "Kick Off (Start)", "Half": "2nd Half"}}]
        pred = _predict_v2(trk, events=events, blocks=[(2, 0, 699)])
        self.assertTrue(np.all(pred[:235] == "D"))
        self.assertTrue(np.all(pred[275:] == "A"))


class V2ControlAndDecoderTests(unittest.TestCase):
    def test_controlled_switches_both_directions_low_latency(self):
        trk = _make_tracking(n=700, owner=0)
        trk.ball[220:460] = trk.players[220:460, 3] + np.array([0.3, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred[20:210] == "H"))
        self.assertTrue(np.all(pred[230:450] == "A"))
        self.assertTrue(np.all(pred[470:] == "H"))
        self.assertLessEqual(abs(int(np.flatnonzero(pred == "A")[0]) - 220), 6)
        self.assertLessEqual(abs(int(np.flatnonzero(pred[400:] == "H")[0] + 400) - 460), 6)

    def test_same_team_pass_past_opponent_bystander_holds_possession(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(25.0, 30.0), away_pos=(45.0, 30.5))
        trk.players[:, 2] = [65.0, 30.0]  # home teammate receiver
        # Fast pass from home 0 (x=25) past away 3 (x=45, 0.5m off line!) to home 2 (x=65) over 25 frames (40m in 1s = 40 m/s? use 1.4s = 35 frames -> 28.5 m/s)
        for t in range(200, 236):
            a = (t - 200) / 35.0
            trk.ball[t] = (1.0 - a) * np.array([25.0, 30.0]) + a * np.array([65.0, 30.0])
        trk.ball[236:] = trk.players[236:, 2] + np.array([0.3, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred == "H"))

    def test_contested_duel_retains_previous_owner_and_deflection_rejected(self):
        trk = _make_tracking(n=600, owner=0, home_pos=(50.0, 30.0), away_pos=(50.8, 30.0))
        # Clear home lead first, then contested duel at 200..235, then home retains.
        trk.players[:200, 3] = [65.0, 30.0]
        trk.players[235:, 3] = [65.0, 30.0]
        # Brief 3-frame deflection near away
        trk.ball[350:353] = trk.players[350:353, 3] + np.array([0.2, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred == "H"))

    def test_genuine_short_possession_survives(self):
        trk = _make_tracking(n=600, owner=0)
        # Away genuinely traps and holds ball for 28 frames (~1.12s).
        trk.ball[220:248] = trk.players[220:248, 3] + np.array([0.2, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred[226:244] == "A"))
        self.assertEqual(pred[320], "H")

    def test_incorrect_and_overlapping_tags_cannot_overpower_geometry(self):
        trk = _make_tracking(n=500, owner=0)
        events = [
            {"code": "BUILD UP", "start_frame": 100, "end_frame": 400, "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
            {"code": "PROGRESSION", "start_frame": 100, "end_frame": 400, "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
            {"code": "BALL IN FINAL THIRD", "start_frame": 100, "end_frame": 400, "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
            {"code": "BALL IN THE BOX", "start_frame": 100, "end_frame": 400, "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
            {"code": "PLAYERS IN THE BOX", "start_frame": 100, "end_frame": 400, "tags": {"Team": "Atletic Club", "Half": "1st Half"}},
        ]
        pred = _predict_v2(trk, events=events)
        self.assertTrue(np.all(pred == "H"))

    def test_duplicate_tracks_and_2frame_glitches_handled(self):
        trk = _make_tracking(n=500, owner=0)
        # Duplicate slot 0 onto slot 1
        trk.players[:, 1] = trk.players[:, 0] + np.array([0.1, 0.05], dtype=np.float32)
        # Inject 2-frame glitch near away
        trk.ball[250:252] = trk.players[250:252, 3] + np.array([0.2, 0.0], dtype=np.float32)
        pred = _predict_v2(trk)
        self.assertTrue(np.all(pred == "H"))
        qf = extract_quality_features(trk.players, trk.ball, trk.teams, FPS, PITCH, Config())
        self.assertFalse(qf["dedup_valid"][100, 1])

    def test_segment_decoder_matches_exhaustive_optimum_time_varying(self):
        rng = np.random.default_rng(7)
        emissions = rng.normal(0.0, 1.0, size=(6, 3))
        acq = np.array([0.0, 0.8, 0.1, 0.9, 0.2, 0.0])
        con = np.array([0.0, 0.2, 0.7, 0.0, 0.5, 0.1])
        stp = np.array([0.0, 0.0, 0.0, 0.9, 0.1, 0.0])
        rst = np.array([0.0, 0.0, 0.0, 0.0, 0.85, 0.0])
        occ = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        costs = build_transition_costs(
            6, 2.0, 1.5, 1.2,
            acquisition_edge=acq, contest_factor=con,
            stop_edge=stp, restart_edge=rst, occluded=occ,
        )
        actual = decode_segments(
            emissions, 2.0, 1.5, 1.2,
            acquisition_edge=acq, contest_factor=con,
            stop_edge=stp, restart_edge=rst, occluded=occ,
        )
        def path_score(p):
            s = sum(emissions[t, p[t]] for t in range(6))
            for t in range(5):
                s -= costs[t, p[t], p[t + 1]]
            return s
        best = max(path_score(p) for p in itertools.product(range(3), repeat=6))
        self.assertAlmostEqual(path_score(actual), best, places=10)


class V2MetamorphicAndInvarianceTests(unittest.TestCase):
    def test_metamorphic_slot_permutation_invariance(self):
        trk = _make_tracking(n=500, owner=0)
        trk.ball[220:] = trk.players[220:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        pred_ref = _predict_v2(trk)

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(trk.teams))
        trk_perm = Tracking(trk.players[:, perm], trk.ball.copy(), trk.teams[perm])
        pred_perm = _predict_v2(trk_perm)
        np.testing.assert_array_equal(pred_ref, pred_perm)

    def test_metamorphic_pitch_reflection_invariance(self):
        trk = _make_tracking(n=500, owner=0)
        trk.ball[220:] = trk.players[220:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        pred_ref = _predict_v2(trk)

        p_mirror = np.array(PITCH, dtype=np.float32)[None, None, :] - trk.players
        b_mirror = np.array(PITCH, dtype=np.float32)[None, :] - trk.ball
        trk_mirror = Tracking(p_mirror, b_mirror, trk.teams.copy())
        pred_mirror = _predict_v2(trk_mirror)
        np.testing.assert_array_equal(pred_ref, pred_mirror)

    def test_metamorphic_event_order_invariance_and_half_split(self):
        trk = _make_tracking(n=700, owner=0)
        trk.ball[400:] = trk.players[400:, 3] + np.array([0.3, 0.0], dtype=np.float32)
        events = [
            {"code": "BUILD UP", "start_frame": 30, "end_frame": 180, "tags": {"Team": "Barcelona", "Half": "1st Half"}},
            {"code": "PROGRESSION", "start_frame": 420, "end_frame": 600, "tags": {"Team": "Atletic Club", "Half": "2nd Half"}},
        ]
        blocks = [(1, 0, 299), (2, 400, 699)]
        p1 = _predict_v2(trk, events=events, blocks=blocks)
        p2 = _predict_v2(trk, events=list(reversed(events)), blocks=blocks)
        np.testing.assert_array_equal(p1, p2)

        # Independent half split matches joint call exactly.
        h1 = _predict_v2(trk, events=events, blocks=[(1, 0, 299)])
        h2 = _predict_v2(trk, events=events, blocks=[(2, 400, 699)])
        np.testing.assert_array_equal(p1[:300], h1[:300])
        np.testing.assert_array_equal(p1[400:], h2[400:])
        self.assertTrue(np.all(p1[300:400] == "X"))

    def test_v2_configs_load_and_determinism(self):
        trk = _make_tracking(n=400, owner=0)
        for cfg_name in ("v2_stoppage", "v2_control", "v2_decoder", "v2"):
            from pathlib import Path
            cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{cfg_name}.json")
            a = infer(trk, [], cfg, blocks=[(1, 0, 399)], fps=FPS, pitch=PITCH)[0]
            b = infer(trk, [], cfg, blocks=[(1, 0, 399)], fps=FPS, pitch=PITCH)[0]
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
