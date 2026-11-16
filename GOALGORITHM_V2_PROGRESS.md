# Goalgorithm v2 — Engineering Progress Log

Start timestamp: 2026-11-16T22:21:11+02:00
Repository HEAD at start: `50833b490b38e2c24b97e5c7ec805093314bc18f`
Target competition metric: `spellF1` (mean one-to-one temporal IoU F1 over thresholds 0.3..0.7), tie-break `macroF1`.

## Checkpoint 1 — Safety Checkpoint, Compliance Audit, and Baseline Verification
- Confirmed CWD, clean Git status, and `Goalgorithm.csv` (199,320 lines).
- Backed up `Goalgorithm.csv` to `artifacts/local_backup/Goalgorithm_v1.csv` and recorded SHA-256 `29e2a65b80dcd6b9e0f5e47e781ca4f97ec9d1e3f91c78a6c550c38a60eed089`.
- Verified Git ignore rules and passed `30/30` existing tests in `.venv`.

## Checkpoint 2 — Codebase Engineering Audit
- Identified v1 failure modes across `tagging.py`, `features.py`, `temporal.py`, and `model.py`.

## Checkpoint 3 — Independent External Data Investigation & Benchmark Suite
- Downloaded `metrica-sports/sample-data` (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`, ~430,000 frames) into Git-ignored `artifacts/external_data/metrica/`.
- Fixed `Sample_Game_3` EPTS `DataFormatSpecification` 22-slot mapping in `scripts/external_benchmark.py`.
- Built two independent benchmark suites (`clean_notags` and `atd_noisy_tags`) evaluated with unchanged `evaluation/possession_metrics.py`.

## Checkpoint 4 — v2 Architecture Implementation, Ablations & Refinements
- Implemented `goalgorithm/quality.py`, `goalgorithm/stoppage.py`, `goalgorithm/control.py`, `goalgorithm/segment_decoder.py`, updated `goalgorithm/config.py`, `goalgorithm/model.py`, `run_goalgorithm.py`, and added versioned configs (`v2_stoppage.json`, `v2_control.json`, `v2_decoder.json`, `v2.json`).
- Added `21` adversarial stoppage, control, exhaustive time-varying DP, and metamorphic tests in `tests/test_v2.py`.
- Ran component isolations and calibrated symmetric weights (`memory_weight=0.55`, `pass_bridge_weight=0.65`, collective high player-speed live-play cue).
- Final external 3-match benchmark results (`clean_notags` | `atd_noisy_tags`):
  - `baseline_50`: `spellF1` 0.3960 | 0.3485
  - `v1`: `spellF1` 0.4610 | 0.4567 (`D_f1` 0.0048 | 0.0191)
  - `v2_decoder`: `spellF1` 0.4664 | 0.4600
  - `v2_control`: `spellF1` 0.4711 | 0.4898
  - `v2_stoppage`: `spellF1` 0.5019 | 0.5256
  - **`v2` (Complete)**: `spellF1` **0.5133 | 0.5473**, `macroF1` **0.6768 | 0.7583**, `D_f1` **0.5191 | 0.6779**, `transF1@25` **0.5512 | 0.5947**, `team_both_alive` **0.8740 | 0.9099**

## Checkpoint 5 — Supervised vs Deterministic Evaluation (Section 11)
- Ran LOMO 3-fold comparison (`scripts/eval_learned_vs_det.py`): compact pure-NumPy linear/ridge model scored `0.5707` vs deterministic `v2` `0.5473`, but weight inspection revealed dependence on simulated missingness (`usable W_D = -0.119`), simulated `tag_prior`, and match-specific intercepts. Retained pure deterministic `v2` to avoid domain-shift risk.

## Checkpoint 6 — Full-Length Synthetic Smoke, Local v2 Candidate Generation, and Validation
- `51/51` unit tests passed (`14.770s`).
- `tests.compare_synthetic` passed (`v2` `spellF1 = 1.0000`, `macroF1 = 0.9974`, `0.021s`).
- `tests.smoke_full --slots 46 --config configs/v2.json` passed twice with identical SHA-256 (`086763f0c2f77c61a96f279110e1b37babc5044bad8d09a5ac3afc7a3eb30967`), `33.66s` / `34.36s`, peak RSS `542.5 MiB`, `40,462` ungraded `X`, validator exit `0`.
- Locally generated `artifacts/local_candidates/Goalgorithm_v2.csv` twice from `./data` without printing or reading any match records/diagnostics: `199,320` lines, identical SHA-256 `fa7f7fbe568553ff6a795c086f8a69c254f9e04281a634811c1720491e249be2`, official validator exit `0` for both `v1` and `v2`, `Goalgorithm.csv` SHA-256 confirmed unchanged (`29e2a65b80dcd6b9e0f5e47e781ca4f97ec9d1e3f91c78a6c550c38a60eed089`).

## Checkpoint 7 — Documentation, Report PDF, and Final Hygiene Verification (Completed)
- Updated `configs/README.md`, `GOALGORITHM_README.md`, `GOALGORITHM_TECHNICAL_REPORT.md`, rebuilt and visually inspected 5-page `Goalgorithm.pdf`, and wrote `GOALGORITHM_V2_ACCEPTANCE.md`.
