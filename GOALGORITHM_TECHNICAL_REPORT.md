# Goalgorithm
## Reproducible Possession Recognition and Event-Driven Segmentation (Final v2)

**Prepared by:** Mahmoud Mohamed Fathy and Youssef Adel  
**Team:** Goalgorithm

Delivery date: November 16, 2026. Competition: More Than a Hack 2027, organized by Metrica Sports, FC Barcelona and Mobile World Capital Barcelona.

**Status: FINAL SUBMISSION v2 PROMOTED AND VALIDATED.** The submitted `Goalgorithm.csv` (`199,319` predictions, SHA-256 `fa7f7fbe568553ff6a795c086f8a69c254f9e04281a634811c1720491e249be2`) is generated deterministically from the authorized tactical bundle using `configs/v2.json` and passes the unmodified official validator (`exit 0`). The prior Phase 1 baseline (`v1`) remains safely backed up at `artifacts/local_backup/Goalgorithm_v1.csv` (`29e2a65b80dcd6b9e0f5e47e781ca4f97ec9d1e3f91c78a6c550c38a60eed089`). Private competition-match `spellF1` remains unknown; no claim of competition-match leaderboard placement is made prior to organizer scoring.

### 1. Task, Ranking Metric, and Architecture

For every frame `0..199318` of the tactical video (25 fps), predict home possession `H`, away possession `A`, dead ball `D`, or abstention `X`. The ranking metric is the official `spellF1`, averaging one-to-one temporal IoU F1 over thresholds `0.3..0.7`, with `macroF1` as the primary tie-break. Evaluation code (`evaluation/possession_metrics.py`, `evaluation/validate_submission.py`, `games.py`) is unmodified.

Goalgorithm preserves the complete Phase 1 (`v1`, `configs/temporal.json`) stack alongside modular Phase 2 (`v2`, `configs/v2.json`) upgrades across four dedicated modules: `goalgorithm/quality.py` (outlier filtering, centered kinematics, slot deduplication), `goalgorithm/stoppage.py` (event-driven stoppage, boundary-exit-to-gap, and restart-kick detection), `goalgorithm/control.py` (multi-frame ball control, duel damping, pass-flight continuity, acquisition edges), and `goalgorithm/segment_decoder.py` (event-modulated 3-state dynamic programming).

| Candidate config | Scope and modules | Primary purpose |
| --- | --- | --- |
| `configs/baseline.json` | Official raw reader + smooth50 | Published floor reproduction |
| `configs/temporal.json` (v1) | Legacy features, tagging, Viterbi | Preserved backup (`Goalgorithm_v1.csv`) |
| `configs/v2.json` (v2) | Quality, stoppage, control, decoder | Final promoted submission (`Goalgorithm.csv`) |

<!-- pagebreak -->

## Data Alignment, Signal Quality, and Ball-Control Reasoning

### 2. Assets, Frame Alignment, and Strict Ingestion

The inference pipeline consumes tactical ATD metadata (`BAR-ATH_tactical_atd.xml`), raw tactical ATD (`BAR-ATH_tactical_atd.txt`), and tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`). Broadcast tracking, video, and homography are excluded because all features operate natively in tactical pitch metres (`105.3 x 68.0 m`).

ATD frame IDs `1..199319` map strictly by `video_frame = frame_id - 1`. Smart Tagging timestamps map via `round(seconds * 25)` on half-open intervals. Inference executes independently on the two graded blocks (`11185..90651` and `110808..190197`); the remaining `40,462` ungraded frames receive `X`. No state, interpolation, or decoding crosses halftime.

### 3. Signal Quality and Centered Kinematics (`quality.py`)

Raw tracking contains single/double-frame teleport spikes, missing observations (`41%` of graded frames in ATD), and duplicate player tracklets. Module `quality.py` applies:

- **1-frame and 2-frame outlier spike masking**: removes isolated teleport bursts whose flanking distance is physically consistent with continuous motion, without deleting valid post-gap returns.

- **Explicit separation of `observed` vs `usable`**: gaps `<= 0.32 s` (`8` frames) with implied speed `<= 45 m/s` are linearly interpolated with initial reliability `0.35`, while `observed` stays `False`. Long gaps remain `NaN` so trajectories are never hallucinated.

- **Centered low-latency velocity**: replaces v1's backward-only difference (`xy[t] - xy[t-lag]`, which introduced systematic restart and kick lag) with centered finite differences and clean one-sided segment-edge fallbacks, plus turn-angle sharpness (`0.5 * (1 - cos_angle)`) to detect kicks and deflections.

- **Chunked same-team slot deduplication**: keeps one representative per overlapping same-team cluster (`< 0.45 m`) in memory-bounded `2048`-frame chunks, preventing duplicate tracks from skewing collective speed or density.

### 4. Multi-Frame Ball Control (`control.py`)

Track slots are never used as persistent player identities. Team support uses slot-agnostic spatial continuity (checking distance from the nearest player at `t` to ANY teammate at `t - 1`), exponential proximity (`2.5 m` scale), and co-motion alignment (`5.5 m/s` velocity difference scale). Crucially, v2 adds:

- **Pass-flight bystander attenuation**: attenuates support when the ball passes rapidly (`> 9 m/s`) past a non-controlling bystander (`d > 1.7 m`, `v_rel > 6.5 m/s`).

- **Contested duel damping**: when both teams are within `2.6 m` with small separation (`contest_factor`), directional deltas are damped by `1 - 0.35 * contest_factor` so possession remains with the prior controller during 50/50 challenges.

- **Same-team pass continuity bridge**: bridges flights/occlusions `<= 2.2 s` connecting two same-team crisp control anchors (`d < 2.0 m`, clear `> 0.55 m` margin) with no opposing control anchor in between.

- **Control acquisition edge (`acquisition_edge`)**: detects sustained `0.44 s` team superiority coupled with ball deceleration/trap or kick/turn sharpness, separating genuine receptions from 2-4 frame deflections.

<!-- pagebreak -->

## Event-Driven Stoppages, Smart Tagging, and Temporal Decoding

### 5. Event-Driven Stoppage & Restart Subsystem (`stoppage.py`)

Testing Goalgorithm v1 on independent public Metrica matches revealed `D_f1` near zero (`0.0048` clean, `0.0191` ATD-degraded) because real stoppages typically exhibit a brief `1-5` frame boundary crossing followed immediately by missing ball tracking (`NaN`) while players reposition at `0.8-1.8 m/s`. Module `stoppage.py` fixes this without ever turning mere uncertainty into `D`:

- **Explicit 3-way evidence separation**: distinguishes positive live play (`usable` inside pitch moving or actively dribbled), positive stoppage evidence (`oob_score`, `exit_gap_score`, `sp_score`, `quiet_score`), and uncertain interior occlusions (assigned a mild negative live-persistence baseline `-0.65`, modulated downward further by collective player speed `> 1.55 m/s`).

- **Boundary-exit-to-gap detection**: when the ball exits `[0, L] x [0, W]` or projects outward across a boundary immediately before a tracking gap (`>= 0.40 s`), `exit_gap_score` supports `D` from the crossing frame until the linked restart or ball reappearance, decaying smoothly after `12 s` if untagged.

- **Set-piece preparation & restart snapping**: parses `SET PIECES` by `Type` (`Throw-in`, `Corner Kick`, `Goal Kick`, `Free Kick`, `Kick Off`), `Half`, and `Team`. Rather than imposing a uniform `D` block, it locates the actual restart kick `t_restart` (ball acceleration `>= 2.2 m/s`, spot displacement `>= 1.5 m`, or reappearance in play), verifies physical stoppage evidence (ignoring contradictory fast-play tags), and walks backward up to `45 s` until encountering sustained live play or an interior controlled ball before a gap.

- **Pre-kickoff & post-goal handling**: handles graded frames prior to a delayed `Kick Off (Start)` (such as pre-kickoff periods inside graded half blocks) and missing/stationary intervals between `GOALS` and subsequent `Kick Off (After Goal)` restarts.

### 6. Bounded Smart Tagging Fusion

Offensive tags (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`, `LONG BALL`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `GOALS`) support the tagged team; `DEFENSIVE` tags support the opponent with weight `0.65`. Correlated overlapping tags are normalized by `max(weight, 1.0)` and contradictory votes cancel. `N/A` tags (including throw-ins) contribute no team bias. When a non-throw-in `SET PIECES` event carries a `Team` tag, `restart_prior` applies a weak decaying prior ONLY after `t_restart` (never before restart while dead).

### 7. Segment-Aware Temporal Decoder (`segment_decoder.py`)

For each graded half independently, the exact dynamic program maximizes over `s_t` in `{0: H, 1: A, 2: D}`:

```text
J(s) = sum_{t=0}^{T-1} E_t(s_t)  -  sum_{t=1}^{T-1} C_t(s_{t-1}, s_t)
```

Emissions combine `1.8 * delta + 0.55 * memory + 0.65 * pass_bridge * (1 - 0.5*rel) + 0.45 * tag_prior * (1 - 0.5*rel) + 0.35 * restart_prior * (1 - 0.5*rel)` scaled by `1 / (1 + 0.7 * max(dead_score, 0))` and penalized by `max(dead_score, 0)` for live states, versus `dead_score` for `D`. Time-varying transition costs snap boundaries to physical events without letting jitter proliferate: `H<->A` (`7.0`) scales by `(1 - 0.45*acquisition_edge) * (1 + 0.35*contest_factor + 0.20*occluded)` clamped to `[0.45, 1.55] * 7.0`; `H/A->D` (`4.5`) reduces up to `55%` at `stop_edge`; and `D->H/A` (`4.5`) reduces up to `65%` at `restart_edge`. No arbitrary minimum possession-duration filter is applied.

<!-- pagebreak -->

## Independent External Evaluation, Ablations, and Engineering QA

### 8. External 3-Match Evaluation (`metrica-sports/sample-data`)

Because BAR-ATH ground truth is private and the public 5-minute window is strictly barred from model selection or tuning, we built an independent external benchmark (`scripts/external_benchmark.py`) using all three public 25 fps matches in `metrica-sports/sample-data` (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`, ~430,000 frames total; `Sample_Game_3` slots mapped strictly via `DataFormatSpecification`). Candidates were scored with the unchanged `evaluation/possession_metrics.py` under two regimes: (1) raw clean tracking with zero tags (`clean_notags`), and (2) realistic ATD-degraded tracking (`~49%` ball visibility with correlated live occlusions, slot permutations, duplicate slots, jitter, partial visible stoppages, and noisy Smart Tagging with missed/mis-tagged phases).

| Model | Clean spellF1 / macroF1 | ATD spellF1 / macroF1 |
| --- | --- | --- |
| `baseline_50` | 0.3960 / 0.7439 | 0.3485 / 0.7325 |
| `v1` (`temporal.json`) | 0.4610 / 0.4922 | 0.4567 / 0.5130 |
| `v2_decoder` (decoder only) | 0.4664 / 0.4928 | 0.4600 / 0.5146 |
| `v2_control` (control only) | 0.4711 / 0.4859 | 0.4898 / 0.5149 |
| `v2_stoppage` (stoppage only) | 0.5019 / 0.6737 | 0.5256 / 0.7531 |
| **`v2` (Complete)** | **0.5133 / 0.6768** | **0.5473 / 0.7583** |

Every v2 module improved both regimes individually, and combined `v2` gained `+0.0906 spellF1`, `+0.2453 macroF1`, `+0.6588 D_f1` (`0.6779` vs `0.0191`), and `+0.1382 transF1@25` (`0.5947` vs `0.4565`) over `v1` on ATD-degraded real matches while keeping `team_both_alive` at `0.9099`.

### 9. Supervised vs Deterministic Model Decision (Section 11)

Leave-One-Match-Out evaluation (`scripts/eval_learned_vs_det.py`) compared a pure-NumPy L2-regularized 3-class linear emission model + v2 decoder against deterministic `v2` across `Sample_Game_1-3` (`atd_noisy_tags`). Although the learned linear model achieved mean `spellF1 = 0.5707` vs `0.5473` on held-out simulated folds, inspecting its weights showed reliance on simulated `usable` (`W_D = -0.119`), simulated `tag_prior` strength, and match-specific intercepts. To avoid domain-shift risk between simulated degradations and real BAR-ATH ATD, we rejected learned weights and retained pure deterministic `v2`.

### 10. Automated Tests, Synthetic Smoke, and Limitations

All `51` unit tests pass (`14.77 s` in `.venv`): the original `30` v1 tests plus `21` new tests (`tests/test_v2.py`) covering boundary-exit gaps, live interior gaps, moving players during stoppages, occluded throw-ins, corners/free kicks, pre-kickoff dead stretches, contradictory tags, contested duels, pass-flight bystanders, duplicate tracks, 2-frame spikes, time-varying exhaustive DP optimality, and metamorphic invariances (slot permutation, pitch reflection, event order, independent half splits).

On the `2,400`-frame synthetic comparison (`tests.compare_synthetic`), `v2` achieves `spellF1 = 1.0000` (`macroF1 = 0.9974`, `0.021 s`). On two full `199,319`-frame, `46`-slot synthetic CLI runs (`tests.smoke_full --slots 46 --config configs/v2.json`), `v2` produced identical SHA-256 bytes (`33.66 s` / `34.36 s`, peak RSS `542.5 MiB`, `40,462` ungraded `X`, validator exit `0`). Known limits remain: untagged central fouls without a visible ball stop or `SET PIECES` context cannot always be distinguished from long live occlusions, and external `Sample_Game_1-3` evaluation uses event-derived GT rather than Metrica's private corrected-tracking GT.

<!-- pagebreak -->

## Reproduction, Compliance, and Submission Summary

### 11. Reproducing the Final Submission (`Goalgorithm.csv`)

Use Python 3.10 or 3.11 with `requirements-goalgorithm.txt` (`numpy==2.2.6`, `pandas==2.3.3`), zero GPU or network calls. Place the authorized bundle in `./data/tactical/` (or set `MTAH_DATA_DIR`), then execute:

```text
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/v2.json --output Goalgorithm.csv
python -X utf8 evaluation/validate_submission.py Goalgorithm.csv
```

To reproduce the archived Phase 1 baseline (`artifacts/local_backup/Goalgorithm_v1.csv`):

```text
python -X utf8 run_goalgorithm.py --data-dir ./data --config configs/temporal.json ^
  --output artifacts/local_backup/Goalgorithm_v1_repro.csv
```

Verify the deterministic engineering suite without competition assets:

```text
python -X utf8 -m compileall -q goalgorithm tests scripts run_goalgorithm.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.compare_synthetic
python -X utf8 -m tests.smoke_full --slots 46 --config configs/v2.json
```

### 12. Compliance and Local Validation Record

- **No competition-match leakage**: No BAR-ATH tracking rows, video frames, Smart Tagging XML contents, derived frame predictions, or match-derived diagnostics entered cloud model context or Git.

- **No tuning on competition data or public 5-minute GT**: Parameters in `configs/v2.json` are principled physical constants verified solely on synthetic fixtures and independent public matches (`metrica-sports/sample-data`).

- **Final submission verification**: `Goalgorithm.csv` (`v2`, `199,319` predictions) passed two independent deterministic local runs with identical SHA-256 (`fa7f7fbe568553ff6a795c086f8a69c254f9e04281a634811c1720491e249be2`) and passed `evaluation/validate_submission.py` (`exit 0`). The Phase 1 submission remains safely archived at `artifacts/local_backup/Goalgorithm_v1.csv` (`29e2a65b80dcd6b9e0f5e47e781ca4f97ec9d1e3f91c78a6c550c38a60eed089`, `validate_submission.py` exit `0`).

- **Promotion basis**: Promoted to `Goalgorithm.csv` based on consistent independent external improvements across 3 real Metrica matches (`+0.0906 spellF1`, `+0.2453 macroF1`, and `+0.6588 D_f1` over `v1` under ATD degradation), `51/51` passing unit/adversarial tests, deterministic `199,319`-frame reproduction, and local validator verification.
