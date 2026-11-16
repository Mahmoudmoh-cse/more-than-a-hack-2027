# Goalgorithm - Phase 1 delivery

Status: **READY EXCEPT FOR LOCAL DATA EXECUTION**. The code and synthetic QA are complete. No `Goalgorithm.csv` from competition data has been generated in this hosted development session. Private-match spellF1 is unknown. The published baseline 0.2433 is not our score.

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

Extract the participant-authorized bundle on your own machine, outside hosted assistant inspection. It must contain:

```text
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The archive in the parent workspace has not been opened, so its contents/completeness have not been verified. Video, homography and broadcast files are not required. Point `--data-dir` at the parent of `tactical/`, not at `tactical/` itself. `MTAH_DATA_DIR` also works when `--data-dir` is omitted.

**Single command to regenerate the submission:**

```bash
python run_goalgorithm.py --data-dir ./data --output Goalgorithm.csv
```

The command parses every ID/slot, performs inference independently in the documented graded halves, writes the two-column UTF-8 CSV and invokes the unmodified official validator. Exit status 0 is required. It validates a temporary candidate before replacing the final path, then validates the final file. Logs do not print tracking records or predictions. A local `Goalgorithm.diagnostics.json` records configuration, label/possession summaries, missing-observation frequency, runtime, peak process memory, dependency versions and CSV hash. Keep this sidecar local; it is Git-ignored.

Verify the exact upload file in your standalone local terminal:

```bash
python -X utf8 evaluation/validate_submission.py Goalgorithm.csv
```

Do not paste the output or match-derived diagnostics into hosted assistants. Run twice to compare hashes if required; inference uses no randomness. The optional `--profile` enables tracemalloc allocation profiling, which slows execution; normal runs already report operating-system peak process memory.

## Fallback and fixed ablations

```bash
python run_goalgorithm.py --data-dir ./data --config configs/baseline.json --output artifacts/Goalgorithm_baseline.csv
python run_goalgorithm.py --data-dir ./data --config configs/geometry.json --output artifacts/Goalgorithm_geometry.csv
python run_goalgorithm.py --data-dir ./data --config configs/tagging.json --output artifacts/Goalgorithm_tagging.csv
```

The default is `configs/temporal.json`. Baseline uses the official raw reader and smooth50, unchanged; geometry and tagging are ablations without a D model. Baseline/geometry need only the two ATD files. Tagging/temporal require the Smart Tagging file, but an empty valid XML is supported. Do not select configurations from public-window scores. No performance ranking on real data has been established.

## Tests without licensed data

```bash
python -X utf8 -m compileall -q goalgorithm tests scripts run_goalgorithm.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.compare_synthetic
python -X utf8 -m tests.smoke_full --slots 46
```

The last command creates a temporary fabricated full-length ATD bundle, runs the real CLI twice, checks identical bytes and graded boundaries, and invokes official validation. It never reads `./data`. Outputs under `artifacts/synthetic/` are **not competition submissions**. The comparison uses a handcrafted synthetic behavior fixture, not a held-out sports dataset. All 30 tests passed; see `GOALGORITHM_ACCEPTANCE.md`.

For scorer installation verification only, the repository's unchanged baselines can be scored with `python -X utf8 evaluation/run_evaluation.py --no-detail`. This was performed during delivery. Never tune or select methods using that public window. The inference command never opens any ground truth or baseline prediction CSV.

## Troubleshooting without disclosing data

- Nonzero exit with missing assets: verify the three exact tactical filenames and root directory above.
- Invalid ATD: IDs must be 1..199319 in order, with exactly the metadata slot count on every row. A raw line has `frame:player;player;...:ball`; each group has x,y,speed. Never pad, reorder, or trim silently. Obtain a valid original bundle if it fails.
- Unknown or reversed team references: check the original metadata locally; first team is home under the documented FIFA/EPTS convention. Do not manually relabel match observations.
- Malformed tagging: verify the original XML and tactical time base; do not hand-correct tag intervals.
- Config errors: use a committed JSON file; unknown keys and nonpositive/nonfinite numeric values fail.
- Output errors: use a writable output directory. A previous accepted output is not replaced unless the new temporary candidate passes validation.
- Windows Unicode errors from official scripts: use `python -X utf8` or `PYTHONUTF8=1`.

Detailed exceptions are intentionally suppressed at the production CLI boundary because parsing exceptions may contain licensed record text. Inspect the implementation and reproduce format issues with synthetic inputs instead of uploading the data.

## Packaging and submission

Code is in `goalgorithm/`; configuration rationale is in `configs/README.md`. The technical document is `GOALGORITHM_TECHNICAL_REPORT.md`, with a five-page `Goalgorithm.pdf` companion. PDF generation is optional and separate from inference:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_report.py
```

Before publishing code, stage only reviewed code/docs/configurations and run `python scripts/check_staged.py`. Never force-add the ignored data, predictions, diagnostics or synthetic outputs. The staged-path check is a guardrail; review staged content as well.

Final participant actions: generate the real CSV locally, require validator exit 0, then submit that CSV, the public code repository URL and the technical PDF through the organizers' emailed form by November 17, 2026. Nothing in this repository submits the form automatically.
