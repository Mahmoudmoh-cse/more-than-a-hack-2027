# Goalgorithm - Phase 1 & Candidate v2

Status: **v1 VALIDATED BASELINE PRESERVED; CANDIDATE v2 IMPLEMENTED & EXTERNALLY BENCHMARKED**.
The v1 submission (`Goalgorithm.csv`, backed up at `artifacts/local_backup/Goalgorithm_v1.csv`) remains untouched and valid. Private competition-match `spellF1` for both v1 and v2 is unknown; the published baseline `0.2433` is a reference floor, not our score.

## Install

Python 3.10 or 3.11 is recommended. Validated here on Windows, Python 3.10.11. The implementation uses portable paths and has a Windows/Linux synthetic CI workflow.

Windows PowerShell:

```powershell
cd more-than-a-hack-2027
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-goalgorithm.txt
python -m pip check
$env:PYTHONUTF8 = "1"
```

Linux:

```bash
cd more-than-a-hack-2027
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-goalgorithm.txt
python -m pip check
export PYTHONUTF8=1
```

## Run on the authorized bundle locally

Place the participant-authorized tactical files in `./data/tactical/` (or set `MTAH_DATA_DIR`):

```text
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

**Reproduce v1 (safety baseline):**

```bash
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/temporal.json --output Goalgorithm.csv
```

**Generate candidate v2 into a separate ignored path:**

```bash
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2.json --output artifacts/local_candidates/Goalgorithm_v2.csv
```

Verify any exact submission candidate in your standalone local terminal:

```bash
python -X utf8 evaluation/validate_submission.py artifacts/local_candidates/Goalgorithm_v2.csv
```

Do not paste validator counts or match-derived `.diagnostics.json` summaries into hosted AI services. Inference is deterministic (`run_goalgorithm.py` uses no randomness and makes no network calls).

## Versioned ablations & candidates

```bash
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/baseline.json --output artifacts/local_candidates/Goalgorithm_baseline.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/geometry.json --output artifacts/local_candidates/Goalgorithm_geometry.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/tagging.json --output artifacts/local_candidates/Goalgorithm_tagging.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2_stoppage.json --output artifacts/local_candidates/Goalgorithm_v2_stoppage.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2_control.json --output artifacts/local_candidates/Goalgorithm_v2_control.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2_decoder.json --output artifacts/local_candidates/Goalgorithm_v2_decoder.csv
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2.json --output artifacts/local_candidates/Goalgorithm_v2.csv
```

## Tests & independent external evaluation

```bash
python -X utf8 -m compileall -q goalgorithm tests scripts run_goalgorithm.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.compare_synthetic
python -X utf8 -m tests.smoke_full --slots 46 --config configs/v2.json
```

Independent 3-match external benchmark (`metrica-sports/sample-data` `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`, placed under `artifacts/external_data/metrica/`):

```bash
python -X utf8 -m scripts.external_benchmark
python -X utf8 -m scripts.eval_learned_vs_det
```

## Packaging and submission

Before committing or pushing code, stage only reviewed code/docs/configurations and verify with:

```bash
python -X utf8 scripts/check_staged.py
```

To rebuild `Goalgorithm.pdf` from `GOALGORITHM_TECHNICAL_REPORT.md` (requires `requirements-report.txt`):

```bash
python -X utf8 scripts/build_report.py
```
