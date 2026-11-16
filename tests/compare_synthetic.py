"""Fixed candidate ablations on a fabricated scenario, not a model-selection dataset."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evaluation import possession_metrics as pm
from goalgorithm.config import Config
from goalgorithm.model import infer
from starter.baseline import write_submission
from tests.fixtures import scene


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'artifacts'/'synthetic'/'comparison'
    results = out/'results'
    results.mkdir(parents=True, exist_ok=True)
    data = scene(n=2400)
    truth = np.full(2400,'H')
    data.ball[650:1250] = data.players[650:1250,2]+[.3,0]
    truth[650:1250] = 'A'
    data.ball[180:260] = np.nan
    data.ball[1000:1100] = np.nan
    data.ball[450:453] = data.players[450:453,2]+[.3,0]
    data.ball[1450:1700] = [45,-1.5]
    truth[1450:1700] = 'D'
    data.ball[1900:2200] = np.nan
    truth[1900:2200] = 'A'
    events = [dict(code='ATTACKING TRANSITION',start_frame=1900,end_frame=2200,
                   tags={'Team':'Atletic Club','Half':'1st Half'})]
    write_submission(str(out/'synthetic_truth.csv'),truth)
    summary = []
    for name in ('baseline', 'geometry', 'tagging', 'temporal', 'v2_stoppage', 'v2_control', 'v2_decoder', 'v2'):
        start = time.perf_counter()
        labels, _ = infer(data,events,replace(Config(),model=name),blocks=[(1,0,2399)])
        duration = time.perf_counter()-start
        write_submission(str(results/f'{name}.csv'),labels)
        scores = pm.evaluate(pm.label_codes(truth),pm.label_codes(labels))
        summary.append({'model':name,'macroF1':scores['macro_f1'],
                        'spellF1':scores['spell_f1_mean'],'runtime_seconds':duration})
    # Exercise official scorer unmodified against a wholly invented truth file.
    subprocess.run([sys.executable,'-X','utf8',str(root/'evaluation'/'run_evaluation.py'),
                    '--gt',str(out/'synthetic_truth.csv'),'--results',str(results),
                    '--out',str(out/'scoring'),'--no-detail'],check=True,cwd=root)
    report = {'scope':'synthetic behavior demonstration only; no match performance inference',
              'models':summary}
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
