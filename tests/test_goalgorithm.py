import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from games import GAME, half_video_blocks, video_to_id, id_to_video
from goalgorithm.config import Config, load_config
from goalgorithm.data import read_tracking, read_metadata
from goalgorithm.features import extract
from goalgorithm.model import infer
from goalgorithm.submission import write_validated
from goalgorithm.tagging import evidence, team_index
from goalgorithm.temporal import viterbi
from starter.load_atd import frame_id_of, video_frame_of
from starter.load_smart_tagging import load as load_tags
from starter.baseline import nearest_player_labels, smooth
from tests.fixtures import scene, bundle

ROOT = Path(__file__).resolve().parents[1]


def predict(data, events=(), config=Config(), blocks=None):
    blocks = [(1, 0, len(data.ball)-1)] if blocks is None else blocks
    return infer(data, events, config, blocks=blocks)[0]


class BehaviorTests(unittest.TestCase):
    def test_team_identification(self):
        for team, label in enumerate('HA'):
            self.assertTrue(np.all(predict(scene(owner=team)) == label))

    def test_controlled_switch(self):
        data = scene()
        data.ball[250:] = data.players[250:, 2]+[.3, 0]
        pred = predict(data)
        self.assertTrue(np.all(pred[30:240] == 'H'))
        self.assertTrue(np.all(pred[270:] == 'A'))
        self.assertLessEqual(abs(np.flatnonzero(pred == 'A')[0]-250), 10)

    def test_brief_deflection(self):
        data = scene()
        data.ball[250:253] = data.players[250:253, 2]+[.3, 0]
        self.assertTrue(np.all(predict(data) == 'H'))

    def test_short_valid_possession_survives(self):
        data = scene()
        data.ball[200:235] = data.players[200:235, 2]+[.3, 0]
        pred = predict(data)
        self.assertTrue(np.all(pred[210:230] == 'A'))
        self.assertEqual(pred[300], 'H')

    def test_missing_ball_inside_possession(self):
        data = scene()
        data.ball[150:220] = np.nan
        self.assertTrue(np.all(predict(data) == 'H'))

    def test_disappearance_at_transition(self):
        data = scene()
        data.ball[180:260] = np.nan
        data.ball[260:] = data.players[260:, 2]+[.3, 0]
        pred = predict(data)
        self.assertTrue(np.all(pred[:170] == 'H'))
        self.assertTrue(np.all(pred[280:] == 'A'))
        self.assertFalse(np.any(pred == 'D'))

    def test_long_gap_is_not_dead(self):
        data = scene(n=1800)
        data.ball[100:1700] = np.nan
        self.assertTrue(np.all(predict(data) == 'H'))

    def test_no_evidence_abstains(self):
        data = scene()
        data.ball[:] = np.nan
        self.assertTrue(np.all(predict(data) == 'X'))

    def test_stationary_live_ball_not_dead_without_evidence(self):
        self.assertTrue(np.all(predict(scene()) == 'H'))

    def test_out_of_bounds_dead(self):
        data = scene()
        data.ball[200:350] = [45, -1.5]
        pred = predict(data)
        self.assertTrue(np.all(pred[220:340] == 'D'))
        self.assertEqual(pred[400], 'H')

    def test_set_piece_and_restart(self):
        data = scene(n=600)
        # Stationary before a moving restart, all fabricated.
        data.ball[250:, 0] += np.arange(350)*.09
        data.players[250:, 0, 0] += np.arange(350)*.09
        events = [dict(code='SET PIECES', start_frame=200, end_frame=290,
                       tags={'Team':'Barcelona', 'Half':'1st Half'})]
        pred = predict(data, events)
        self.assertTrue(np.all(pred[100:240] == 'D'))
        self.assertTrue(np.all(pred[280:500] == 'H'))

    def test_set_piece_alone_not_dead(self):
        data = scene()
        data.ball[:, 0] += np.arange(500)*.08
        data.players[:, 0, 0] += np.arange(500)*.08
        event = dict(code='SET PIECES', start_frame=100, end_frame=400, tags={'Team':'N/A'})
        self.assertFalse(np.any(predict(data, [event]) == 'D'))

    def test_halves_reset_and_ungraded_x(self):
        data = scene(n=600)
        data.ball[350:] = data.players[350:, 2]+[.3, 0]
        blocks = [(1, 50, 249), (2, 350, 549)]
        pred = predict(data, blocks=blocks)
        self.assertTrue(np.all(pred[50:250] == 'H'))
        self.assertTrue(np.all(pred[350:550] == 'A'))
        self.assertTrue(np.all(pred[250:350] == 'X'))
        self.assertEqual(pred[0], 'X')
        self.assertEqual(pred[-1], 'X')

    def test_half_without_evidence_does_not_inherit(self):
        data = scene()
        data.ball[300:] = np.nan
        pred = predict(data, blocks=[(1, 0, 199), (2, 300, 499)])
        self.assertTrue(np.all(pred[300:] == 'X'))

    def test_slot_permutation_does_not_change_team(self):
        data = scene()
        data.players[250:, [0, 1]] = data.players[250:, [1, 0]]
        self.assertTrue(np.all(predict(data) == 'H'))
        features = extract(data.players, data.ball, data.teams, 25, (105.3, 68), Config())
        self.assertGreater(features['slot_jump_count'], 0)

    def test_determinism_all_candidates(self):
        data = scene()
        for model in ('baseline', 'geometry', 'tagging', 'temporal'):
            c = replace(Config(), model=model)
            np.testing.assert_array_equal(predict(data, config=c), predict(data, config=c))

    def test_interpolation_bounded(self):
        data = scene()
        data.ball[100:104] = np.nan
        data.ball[200:240] = np.nan
        f = extract(data.players, data.ball, data.teams, 25, (105.3, 68), Config())
        self.assertTrue(np.isfinite(f['ball'][100:104]).all())
        self.assertTrue((f['reliability'][100:104] < .36).all())
        self.assertTrue(np.isnan(f['ball'][200:240]).all())

    def test_fast_impossible_gap_not_interpolated(self):
        data = scene()
        data.ball[100:104] = np.nan
        data.ball[104:] = [100, 60]
        f = extract(data.players, data.ball, data.teams, 25, (105.3, 68), Config())
        self.assertTrue(np.isnan(f['ball'][100:104]).all())

    def test_viterbi_matches_exhaustive_optimum(self):
        import itertools
        emissions = np.array([[.2, .6, -.3], [.3, -.4, .5], [.8, .1, -.3], [.2, 1., .1]])
        penalty = np.array([[0, .7, .4], [.7, 0, .4], [.4, .4, 0]])
        def score(path):
            return sum(emissions[i, p] for i, p in enumerate(path))-sum(penalty[a,b] for a,b in zip(path, path[1:]))
        actual = viterbi(emissions, .7, .4)
        self.assertAlmostEqual(score(actual), max(score(p) for p in itertools.product(range(3), repeat=4)))


class AlignmentAndInputTests(unittest.TestCase):
    def test_official_boundaries_and_id_roundtrip(self):
        self.assertEqual(half_video_blocks(GAME), [(1,11185,90651),(2,110808,190197)])
        for _, a, b in half_video_blocks(GAME):
            for frame in (a, b):
                self.assertEqual(id_to_video(GAME, video_to_id(GAME, frame)), frame)
                self.assertEqual(video_frame_of(frame_id_of(frame)), frame)
        self.assertIsNone(video_to_id(GAME, 90652))
        self.assertIsNone(video_to_id(GAME, 110807))
        self.assertIsNone(video_to_id(GAME, 190198))

    def test_team_tag_na_aliases_conflicts(self):
        names = ('Barcelona', 'Atletic Club')
        self.assertEqual(team_index('Atletic Club', names), 1)
        self.assertIsNone(team_index('N/A', names))
        events = [dict(code='BUILD UP', start_frame=2, end_frame=8, tags={'Team':'Barcelona'}),
                  dict(code='PROGRESSION', start_frame=2, end_frame=8, tags={'Team':'Atletic Club'}),
                  dict(code=None, start_frame='bad', end_frame=8, tags={})]
        prior, _, _ = evidence(events, 0, 10, 1, names, 25, Config())
        self.assertTrue(np.all(prior == 0))

    def test_defensive_tags_invert_and_half_mismatch_ignored(self):
        event = dict(code='DEFENSIVE TRANSITION', start_frame=2, end_frame=8, tags={'Team':'Barcelona','Half':'1st Half'})
        prior, _, _ = evidence([event], 0, 10, 1, ('Barcelona','Atletic Club'), 25, Config())
        self.assertTrue((prior[2:8] < 0).all())
        other, _, _ = evidence([event], 0, 10, 2, ('Barcelona','Atletic Club'), 25, Config())
        self.assertTrue((other == 0).all())

    def test_metadata_and_metric_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle(tmp, 2)
            meta = folder/'BAR-ATH_tactical_atd.xml'
            raw = folder/'BAR-ATH_tactical_atd.txt'
            data = read_tracking(meta, raw, 2, (105.3, 68))
            np.testing.assert_allclose(data.players[0,0], [42.12,30.6], atol=1e-4)
            np.testing.assert_array_equal(data.teams, [0,0,1,1])
            meta.write_text('<Root><Team id="h"/><Team id="a"/><Player teamId="unknown"/></Root>')
            with self.assertRaises(ValueError):
                read_metadata(meta)

    def test_nonconsecutive_duplicate_missing_and_truncated_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle(tmp, 4)
            meta, raw = folder/'BAR-ATH_tactical_atd.xml', folder/'BAR-ATH_tactical_atd.txt'
            original = raw.read_text().splitlines()
            for ids in ([1,3,3,4], [1,2,4,4], [0,1,2,3]):
                raw.write_text('\n'.join(str(i)+line[line.index(':'):] for i,line in zip(ids,original)))
                with self.assertRaises(ValueError):
                    read_tracking(meta, raw, 4, (105.3,68))
            raw.write_text('\n'.join(original[:3]))
            with self.assertRaises(ValueError):
                read_tracking(meta, raw, 4, (105.3,68))

    def test_invalid_coordinates_masked_and_slots_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle(tmp, 1)
            meta, raw = folder/'BAR-ATH_tactical_atd.xml', folder/'BAR-ATH_tactical_atd.txt'
            raw.write_text('1:50,0.5,0;NaN,0.5,0;0.6,0.5,0;0.7,0.5,0:inf,0,0\n')
            data = read_tracking(meta, raw, 1, (105.3,68))
            self.assertEqual(data.invalid_coordinates, 3)
            self.assertTrue(np.isnan(data.ball).all())
            raw.write_text('1:0.4,0.5,0:0.4,0.5,0\n')
            with self.assertRaises(ValueError):
                read_tracking(meta, raw, 1, (105.3,68))

    def test_smart_tagging_seconds_and_half_open_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle(tmp, 1)
            path = folder/'BAR-ATH_tactical_smart_tagging.xml'
            path.write_text('<file><instance><start>0.08</start><end>0.20</end><code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label></instance></file>')
            events = load_tags(data_dir=tmp)
            self.assertEqual((events[0]['start_frame'], events[0]['end_frame']), (2,5))
            prior, _, _ = evidence(events,0,10,1,('Barcelona','Atletic Club'),25,Config())
            self.assertEqual(prior[4], 1)
            self.assertEqual(prior[5], 0)

    def test_baseline_matches_official_on_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle(tmp, 1100)
            raw, meta = folder/'BAR-ATH_tactical_atd.txt', folder/'BAR-ATH_tactical_atd.xml'
            data = read_tracking(meta, raw, 1100, (105.3,68))
            expected = smooth(nearest_player_labels(raw,data.teams,1100,(105.3,68)),50)
            np.testing.assert_array_equal(predict(data,config=replace(Config(),model='baseline')), expected)

    def test_versioned_default_and_invalid_configuration(self):
        self.assertEqual(Config(), load_config(ROOT/'configs/temporal.json'))
        with self.assertRaises(ValueError):
            replace(Config(), switch_cost=float('nan')).validate()

    def test_full_length_csv_and_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'synthetic_a.csv', Path(tmp)/'synthetic_b.csv'
            labels = np.resize(np.array(list('HADX')),199319)
            write_validated(a,labels)
            write_validated(b,labels)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(a.read_text().splitlines()[0], 'frame,label')
            self.assertEqual(len(a.read_text().splitlines()), 199320)
            original = a.read_bytes()
            with self.assertRaises(ValueError):
                write_validated(a, ['Q'])
            self.assertEqual(a.read_bytes(), original)

    def test_missing_data_cli_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable,str(ROOT/'run_goalgorithm.py'),'--data-dir',tmp,
                                     '--output',str(Path(tmp)/'out.csv')],capture_output=True)
            self.assertEqual(result.returncode,1)
            self.assertNotIn(b'Traceback',result.stderr)
            self.assertFalse((Path(tmp)/'out.csv').exists())


if __name__ == '__main__':
    unittest.main()
