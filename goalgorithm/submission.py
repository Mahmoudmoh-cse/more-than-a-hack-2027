"""Atomic CSV writing with the unmodified official validator as the acceptance gate."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def write_validated(path, labels, expected_rows=199319):
    path = Path(path)
    if len(labels) != expected_rows or not set(np.asarray(labels).tolist()) <= set("HADX"):
        raise ValueError("submission length or alphabet invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".goalgorithm-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("frame,label\n")
            handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
        validator = Path(__file__).resolve().parents[1]/"evaluation"/"validate_submission.py"
        checked = subprocess.run([sys.executable, "-X", "utf8", str(validator), temporary,
                                  "--rows", str(expected_rows)], capture_output=True)
        if checked.returncode:
            raise ValueError("Official submission validator rejected the candidate")
        os.replace(temporary, path)
        # Validate the exact final pathname too. Never echo validator label summaries in
        # automated hosted sessions; the participant can run it locally when desired.
        checked = subprocess.run([sys.executable, "-X", "utf8", str(validator), str(path),
                                  "--rows", str(expected_rows)], capture_output=True)
        if checked.returncode:
            raise ValueError("Official submission validator rejected the final file")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def label_summary(labels, blocks, fps):
    from .features import runs
    durations = []
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        graded[first:last+1] = True
        part = labels[first:last+1]
        for label in ("H", "A"):
            durations.extend((end-start)/fps for start, end in runs(part == label))
    return {"n_possessions_raw": len(durations),
            "mean_possession_seconds_raw": float(np.mean(durations)) if durations else 0.,
            "x_percent_all": float(100*np.mean(labels == "X")),
            "x_percent_graded": float(100*np.mean(labels[graded] == "X")) if graded.any() else 0.,
            "label_counts": {label: int(np.sum(labels == label)) for label in "HADX"}}
