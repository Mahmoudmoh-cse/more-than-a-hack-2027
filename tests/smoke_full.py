"""Full-sized synthetic CLI test. Never opens competition assets or ground truth.

python -X utf8 -m tests.smoke_full --slots 46
Outputs under artifacts/synthetic/ are fabricated and MUST NOT be submitted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from games import GAME, GAMES
from games import half_video_blocks
from evaluation.possession_metrics import read_label_csv
import numpy as np
from tests.fixtures import bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots', type=int, default=4)
    args = parser.parse_args()
    if args.slots < 2 or args.slots % 2:
        parser.error('slots must be an even positive number >= 2')
    root = Path(__file__).resolve().parents[1]
    output = root/'artifacts'/'synthetic'
    output.mkdir(parents=True, exist_ok=True)
    (output/'DO_NOT_SUBMIT.txt').write_text('All files here are synthetic QA fixtures, not competition predictions.\n')
    n = GAMES[GAME]['n_video_frames']
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='goalgorithm-synthetic-') as tmp:
        bundle(tmp, n, args.slots)
        hashes = []
        for run in (1, 2):
            path = output/f'synthetic_smoke_{run}.csv'
            subprocess.run([sys.executable, '-X', 'utf8', str(root/'run_goalgorithm.py'),
                            '--data-dir', tmp, '--output', str(path)], check=True, cwd=root)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if hashes[0] != hashes[1]:
            raise AssertionError('repeated full inference is not byte deterministic')
        table = read_label_csv(output/'synthetic_smoke_1.csv')
        active = np.zeros(n, dtype=bool)
        for _, first, last in half_video_blocks(GAME):
            active[first:last+1] = True
        assert (table['label'].to_numpy()[~active] == 'X').all()
        assert (table['label'].to_numpy()[active] != 'X').all()
        assert int((~active).sum()) == 40462
        subprocess.run([sys.executable, '-X', 'utf8', str(root/'evaluation'/'validate_submission.py'),
                        str(output/'synthetic_smoke_1.csv')], check=True, cwd=root)
    report = {'data':'entirely synthetic; not a competition result', 'frames':n, 'slots':args.slots,
              'runs':2, 'sha256_equal':True, 'sha256':hashes[0],
              'total_seconds_including_generation':time.perf_counter()-start,
              'run_diagnostics':[json.loads((output/f'synthetic_smoke_{i}.diagnostics.json').read_text()) for i in (1,2)]}
    (output/'smoke_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'run_diagnostics'},indent=2))
    print('PASS: full synthetic CLI, official validation and identical repeated bytes.')


if __name__ == '__main__':
    main()
