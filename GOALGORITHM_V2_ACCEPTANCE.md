# Goalgorithm v2 — Final Acceptance Gates

Date: 2026-11-16. Verified with `F:\محمود محمد فتحي\learning\Barcelona_Hack2027\.venv\Scripts\python.exe` (Python 3.10.11, NumPy 2.2.6, pandas 2.3.3).

| Gate | Status | Verified Evidence |
| --- | --- | --- |
| Gate 1 — Integrity | PASS | Original `Goalgorithm.csv` (199,320 lines) backed up to `artifacts/local_backup/Goalgorithm_v1.csv`; SHA-256 `29e2a65b80dcd6b9e0f5e47e781ca4f97ec9d1e3f91c78a6c550c38a60eed089` recorded locally and verified unchanged after all edits. |
| Gate 2 — Source quality | PASS | `compileall -q goalgorithm tests scripts run_goalgorithm.py` passed with exit 0; modular v2 modules (`quality.py`, `stoppage.py`, `control.py`, `segment_decoder.py`) import cleanly without circular or external dependencies. |
| Gate 3 — Tests | PASS | `51/51` unit tests passed in `14.770s` (`30/30` original `test_goalgorithm.py` + `21/21` new `test_v2.py` adversarial stoppage, control, exhaustive DP, and metamorphic tests). `tests.compare_synthetic` achieved `spellF1 = 1.0000` for `v2` (`0.021s`). |
| Gate 4 — Correctness | PASS | Full `199,319`-frame `46`-slot synthetic CLI (`tests.smoke_full --slots 46 --config configs/v2.json`) passed with `40,462` ungraded `X`, `0` graded `X`, valid alphabet `{H, A, D, X}`, and independent per-half decoding. |
| Gate 5 — Reproducibility | PASS | Repeated synthetic CLI runs (`smoke_full`) produced identical SHA-256 (`086763f0c2f77c61a96f279110e1b37babc5044bad8d09a5ac3afc7a3eb30967`). Repeated local competition runs of `configs/v2.json` produced identical SHA-256 (`fa7f7fbe568553ff6a795c086f8a69c254f9e04281a634811c1720491e249be2`). |
| Gate 6 — Official validation | PASS | Unmodified `evaluation/validate_submission.py` exited `0` on both `Goalgorithm.csv` (`v1`) and `artifacts/local_candidates/Goalgorithm_v2.csv` (`v2`, 199,320 lines). |
| Gate 7 — Competition compliance | PASS | Zero competition tracking rows, video frames, Smart Tagging XML contents, derived frame predictions, or match-derived diagnostics read into model context or committed; `.gitignore` verified via `git check-ignore`; `evaluation/`, `starter/`, `games.py` untouched. |
| Gate 8 — Performance evidence | PASS (EXTERNAL) / UNVERIFIED (PRIVATE) | Independent 3-match external benchmark on `metrica-sports/sample-data` (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`) using unchanged `possession_metrics.py` shows `v2` improves over `v1` in both clean (`spellF1` `0.5133` vs `0.4610`) and ATD-degraded + noisy-tag regimes (`spellF1` `0.5473` vs `0.4567`, `macroF1` `0.7583` vs `0.5130`, `D_f1` `0.6779` vs `0.0191`, `transF1@25` `0.5947` vs `0.4565`). Private competition-match `spellF1` remains unmeasured until organizer grading. |
| Gate 9 — Documentation | PASS | Updated `configs/README.md`, `GOALGORITHM_README.md`, `GOALGORITHM_TECHNICAL_REPORT.md`, rebuilt 5-page `Goalgorithm.pdf` (visually inspected all 5 pages), `GOALGORITHM_V2_PROGRESS.md`, and `GOALGORITHM_V2_ACCEPTANCE.md`. |
| Gate 10 — Submission safety | PASS | Primary `Goalgorithm.csv` (`v1`) remains valid and untouched; candidate `v2` sits safely in Git-ignored `artifacts/local_candidates/Goalgorithm_v2.csv`. |

## Promotion Recommendation
- **Evidence basis**: Every v2 subsystem (`v2_stoppage`, `v2_control`, `v2_decoder`) independently beat `v1` across 3 independent real Metrica matches under ATD degradation, and full `v2` gained `+0.0906 spellF1`, `+0.2453 macroF1`, and `+0.6588 D_f1` over `v1` (`+0.1988 spellF1` over `baseline_50`) with identical determinism and `5.06s`/match runtime.
- **Actionable safety**: Because `v2`'s competition-match label summary and possession counts were intentionally shielded from hosted model context, the participant should execute `python -X utf8 evaluation/validate_submission.py artifacts/local_candidates/Goalgorithm_v2.csv` locally to sanity-check its non-degenerate match-level possession counts before deciding whether to copy `artifacts/local_candidates/Goalgorithm_v2.csv` over `Goalgorithm.csv` as the final upload (retaining `artifacts/local_backup/Goalgorithm_v1.csv` as immediate fallback).

