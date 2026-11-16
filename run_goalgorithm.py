"""Run locally on the authorized bundle. No network, ground truth, or record logging."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import tracemalloc

from games import GAME, GAMES, asset_paths, half_video_blocks
from goalgorithm.config import load_config
from goalgorithm.diagnostics import peak_rss_mib
from goalgorithm.data import load_inputs
from goalgorithm.model import infer
from goalgorithm.submission import label_summary, write_validated


def run(args):
    config = load_config(args.config)
    start = time.perf_counter()
    if args.profile:
        tracemalloc.start()
    tracking, events = load_inputs(args.data_dir, need_tags=config.model in {"tagging", "temporal"})
    if config.model == "baseline":
        from starter.baseline import nearest_player_labels, smooth
        paths = asset_paths(data_dir=args.data_dir)
        codes = nearest_player_labels(paths["atd_raw"], tracking.teams,
                                      GAMES[GAME]["n_video_frames"], GAMES[GAME]["pitch"])
        labels = smooth(codes, config.baseline_window)
        diagnostics = {"model": "baseline", "implementation": "unmodified starter reader and smoothing"}
    else:
        labels, diagnostics = infer(tracking, events, config)
    write_validated(args.output, labels)
    diagnostics.update(label_summary(labels, half_video_blocks(GAME), GAMES[GAME]["fps"]))
    diagnostics["runtime_seconds"] = time.perf_counter()-start
    diagnostics["peak_process_rss_mib"] = peak_rss_mib()
    diagnostics["tracemalloc_peak_mib"] = tracemalloc.get_traced_memory()[1]/2**20 if args.profile else None
    if args.profile:
        tracemalloc.stop()
    diagnostics["config"] = asdict(config)
    diagnostics["python"] = platform.python_version()
    diagnostics["dependencies"] = {name: importlib.metadata.version(name) for name in ("numpy", "pandas")}
    diagnostics["csv_sha256"] = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    diagnostics["official_validator_exit_status"] = 0
    diagnostics["private_match_spellF1"] = None
    diagnostics["data_policy"] = "Local-only aggregate diagnostics. Do not publish or paste into hosted services."
    # Sidecar is deliberately kept beside the submission and excluded from Git.
    report = Path(args.output).with_suffix(".diagnostics.json")
    report.write_text(json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")
    print("SUCCESS: CSV written; official validator passed. Diagnostics remain local.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="bundle root; defaults to MTAH_DATA_DIR or ./data")
    parser.add_argument("--output", default="Goalgorithm.csv")
    parser.add_argument("--config", default=None, help="versioned JSON; default is temporal fusion")
    parser.add_argument("--profile", action="store_true", help="measure traced allocation peak (slower)")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        # Parsing exceptions can contain restricted record substrings. Never print them.
        print("FAILED: no accepted run. Check tactical filenames, strict frame/slot schema, "
              "configuration and output permissions; see GOALGORITHM_README.md.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
