"""Entirely invented scenes; no competition observations or labels."""

from pathlib import Path
import numpy as np

from goalgorithm.data import Tracking


def scene(n=500, owner=0):
    players = np.zeros((n, 4, 2), dtype=np.float32)
    players[:, 0] = [40, 30]
    players[:, 1] = [35, 25]
    players[:, 2] = [65, 30]
    players[:, 3] = [70, 25]
    ball = players[:, owner*2].copy()+[0.3, 0.]
    return Tracking(players, ball, np.array([0, 0, 1, 1], dtype=np.int8))


def bundle(folder, n, slots=4):
    """Generate full-length fabricated ATD and tag XML to test real ingestion/CLI."""
    folder = Path(folder)/"tactical"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder/"BAR-ATH_tactical"
    metadata = '<Root><Team id="home"><Name>Barcelona</Name></Team><Team id="away"><Name>Atletic Club</Name></Team>'
    metadata += ''.join(f'<Player teamId="{"home" if i < slots//2 else "away"}"/>' for i in range(slots))
    Path(str(stem)+"_atd.xml").write_text(metadata+'</Root>', encoding="utf-8")
    home = [f'{0.40+(i%3)*0.01},0.45,0' for i in range(slots//2)]
    away = [f'{0.65+(i%3)*0.01},0.45,0' for i in range(slots-slots//2)]
    groups = ';'.join(home+away)
    with Path(str(stem)+"_atd.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for i in range(n):
            phase = (i//250)%4
            ball = '0.403,0.45,0' if phase < 2 else '0.653,0.45,0'
            if i%250 in range(100, 106):
                ball = 'NaN,NaN,NaN'
            handle.write(f'{i+1}:{groups}:{ball}\n')
    Path(str(stem)+"_smart_tagging.xml").write_text('<file><ALL_INSTANCES/></file>', encoding="utf-8")
    return folder
