# Goalgorithm
## Phase 1 - Reproducible possession recognition

Delivery date: November 16, 2026. Competition: More Than a Hack 2027, organized by Metrica Sports, FC Barcelona and Mobile World Capital Barcelona.

**Status: READY EXCEPT FOR LOCAL DATA EXECUTION.** This document describes an implemented and synthetically verified system. No licensed-match CSV was generated in this development session. Goalgorithm's private-match spellF1 is unknown; no improvement over the published baseline has been established.

### 1. Task and architecture

For every tactical-video frame, predict home possession H, away possession A, dead ball D, or abstention X. The output contains 199,319 rows, indexed 0 through 199318. The ranking criterion is the official spellF1, averaged over possession overlap thresholds 0.3 to 0.7; macroF1 breaks ties. The evaluation code is unchanged.

The primary method is an interpretable CPU pipeline: strict ATD ingestion, metric geometry and reliability features, bounded Smart Tagging evidence, independent stoppage evidence, then a three-state Viterbi decoder applied separately to each graded half. No neural network, training procedure, external inference service or GPU is required.

Four fixed candidates are provided: A, the official smooth50 baseline; B, enhanced geometry and controller memory; C, geometry plus tactical tags; D, temporal fusion with explicit stoppage detection. D is the engineering default, not the winner of public-ground-truth model selection. B and C are ablations without a dead-ball detector.

### 2. Assets and alignment

The default consumes tactical ATD metadata, raw ATD tracking and tactical Smart Tagging. Metadata supplies slot-to-team membership. Smart Tagging supplies automatic tactical intervals. Both are noisy, uncorrected inputs. The local authorized bundle remains the only match input at inference.

Broadcast tracking is excluded because its time base is independent. Video and homography are excluded because all implemented features operate in pitch coordinates. Their omission reduces dependencies and avoids unnecessary restricted-data processing; they were not experimentally tested and rejected on performance grounds.

The delivered ATD uses frame IDs 1..199319, mapped by video_frame = frame_id - 1. The separate ground-truth IdFrame offsets, 1702057 and 1702058, are used only through the official graded-boundary helpers. They must not be applied to delivered ATD. Smart Tagging timestamps use round(seconds * 25), with half-open intervals.

Graded blocks are inclusive 11185..90651 and 110808..190197. The second block starts before the second kickoff. All other 40,462 frames receive X in the primary model. No filtering, interpolation or decoding crosses a half boundary.

<!-- pagebreak -->

## Features, controller inference and missing observations

### 3. Strict ingestion and physical features

The raw reader streams text into compact float32 arrays. Every record ID must match its expected position and every row must match the metadata slot count. Unknown slot team references, reversed known team names, missing records, duplicates and malformed groups are rejected. Production errors never echo input records.

Normalized x,y positions scale independently by 105.3 and 68.0 metres. Finite coordinates in a small margin outside the pitch remain available for out-of-bounds evidence; grossly invalid, infinite and partial coordinates become missing observations. Nothing is silently clipped onto a boundary.

Ball features include visibility, displacement-derived velocity, acceleration, observation reliability, missing-interval duration and pitch-boundary proximity. Velocity uses a 0.20-second window. Isolated jumps are removed; implausible intermediate motion invalidates derivatives. Fast motion and large acceleration reduce proximity confidence because a fast or aerial ball is not necessarily controlled by the nearest player.

Team features include nearest-player distance, nearby-player density, relative player-ball velocity, geometric controller continuity, team centroids and spread, and collective player speed. The controller score uses nearest distance, relative motion and continuity. Density and spatial configuration are extracted for diagnostics and future independent validation; they are not given unvalidated direct ownership weights.

### 4. Controlled possession and track slots

For each team, control support decreases exponentially with nearest-player distance on a 2.5 m scale. Co-motion and locally continuous controller position increase support modestly. The home-away support difference is weighted by ball reliability. A weak decaying historical controller score preserves ownership during passes and loose-ball intervals.

Track slots are never treated as persistent player identities. Velocity is invalidated across missing observations or jumps beyond a 13 m/s player bound plus 0.6 m jitter allowance. Controller continuity compares positions rather than identity numbers. This handles obvious slot exchanges but cannot detect every geometrically plausible identity swap.

### 5. Occlusion policy

Offline interpolation is restricted to gaps of at most 0.32 seconds, with observed endpoints and implied speed no greater than 45 m/s. Interpolated locations receive initial reliability 0.35, lower than observed positions. Long gaps are never filled with fabricated reliable ball positions.

Across larger gaps, the model combines decaying previous controller evidence, bounded tactical priors and temporal continuity. Bidirectional Viterbi uses observations on both sides to locate a plausible state sequence. It cannot recover the true instant of a hidden transition without evidence; this is a limitation rather than a reason to assert precise timing.

Missing ball alone does not support D. With no ownership or stoppage evidence anywhere in a half, the model abstains for that half. Otherwise it emits H/A/D through uncertain stretches to avoid the macroF1 penalty of indiscriminate X predictions. This is a persistence policy, not calibrated confidence.

<!-- pagebreak -->

## Tactical evidence, stoppages and temporal decoding

### 6. Smart Tagging as noisy evidence

Build Up, Progression, Creating Chances, Attacking Transition, Long Ball and attacking-box phases support the tagged team. Defensive phases weakly support its opponent. Barcelona and the documented Atletic Club spelling are recognized, as are explicit aliases. N/A, missing and unknown teams contribute no ownership evidence.

Overlapping votes are normalized and capped; correlated tags cannot create unlimited evidence. Contradictory votes cancel. Half tags are checked, and intervals are clipped to the relevant half. Invalid event ranges are ignored. Malformed XML is rejected by the reader rather than silently guessed.

Set-piece intervals supply restart context only. They do not directly label D, do not identify an exact restart frame and do not impose a team owner across the whole interval. Context extends at most eight seconds before the event within the half.

### 7. Independent dead-ball logic

The detector supports D when a reliably observed ball stays outside the pitch margin for at least 0.5 seconds; when a stationary observed ball persists for 1.5 seconds with set-piece context; or when the ball is stationary near a boundary for four seconds with independently low collective player motion. A missing ball can contribute to a stoppage only in conjunction with restart context and at least two seconds of low player motion.

Positive stoppage evidence lowers both live-team emission scores. This prevents a nearby player standing over a dead ball from overriding the D detector. Observed ball movement removes the stationary condition, allowing the decoder to return to a live state after a restart. These rules cannot infer every whistle and may miss untagged central stoppages or confuse unusual slow play with a restart.

### 8. Temporal objective

Viterbi maximizes the sum of per-frame H/A/D evidence minus transition costs. Scores are engineering evidence values, not calibrated probabilities. A team switch normally costs 7 units; a transition to or from D costs 5. Strong geometric recovery evidence can reduce the team-switch cost by at most 25 percent, never to zero.

This creates an evidence-dependent duration preference. A brief deflection cannot normally pay the cost of switching away and back, while a short interval of strong controlled possession can survive. There is no blanket minimum possession duration and no forced target possession count or share. A brute-force enumeration test checks the decoder's optimum on a short chain.

Loose or contested balls retain the previous controlled state unless sufficient contrary evidence accumulates. Restart boundaries can split two possessions of the same team through D. Whole-half resetting prevents propagation across halftime. X is reserved for ungraded frames or a wholly uninformative half; the official scorer's within-half filling of X is unchanged.

### 9. Training and configuration provenance

There are no training datasets, pretrained weights, manually annotated match labels or external event data. Numerical parameters are fixed engineering choices documented in configs/README.md. Synthetic fixtures test intended behavior; they do not constitute independent validation. The public five-minute ground truth was used only to verify the unchanged baseline scorer installation, never to train, tune or select Goalgorithm configurations.

<!-- pagebreak -->

## Execution evidence and limitations

### 10. Actual tests and measured results

Thirty automated tests passed on Windows/Python 3.10.11. They cover team identification, controlled switches, brief deflections, short valid possessions, short/long occlusions, hidden transitions, stoppages, restarts, malformed inputs, metric coordinate scaling, both half boundaries, tag conflicts/N/A, slot discontinuities, deterministic inference, full-length CSV generation and Viterbi optimality. The pinned dependencies were installed into a clean virtual environment and pip check passed.

Two complete CLI executions used an entirely invented 199,319-frame, 46-slot ATD bundle. Both passed the unchanged official validator and produced identical CSV bytes. Runtime including input loading and validation was 26.38 and 25.41 seconds. Peak process memory was 285.27 and 285.11 MiB, measured with Windows process counters. This fixture has simple repeated geometry; real input runtime may differ.

Synthetic full-run diagnostics: 320 predicted possessions, mean duration 19.857 seconds, 0 percent X within graded blocks, and 20.300 percent X overall (the ungraded frames). Missing ball frequency was approximately 2.40 percent per half. These figures describe invented data, not the competition match.

The official scorer was also run, unchanged, against a separate 2,400-frame handcrafted synthetic fixture containing switches, deflection, occlusion, an out-of-bounds stoppage and a tagged hidden transition:

| Fixed method | Synthetic macroF1 | Synthetic spellF1 |
| --- | --- | --- |
| A: official baseline | 0.5591 | 0.6000 |
| B: geometry | 0.5390 | 0.3273 |
| C: geometry + tags | 0.6354 | 0.6154 |
| D: temporal fusion | 1.0000 | 1.0000 |

These numbers demonstrate behavior on a deliberately simple fixture. They are not an estimate of real-match performance, not held-out validation, and not evidence that the default beats the competition baseline.

For installation verification, the shipped official smooth50 predictions reproduced public-window macroF1 0.6207 and spellF1 0.3256. The organizer-published full-match smooth50 reference is spellF1 0.2433. Both belong to the official baseline. Goalgorithm has no measured competition-match score.

### 11. Known limits

The system has not been run against licensed inputs here. Schema compatibility is based on the public starter implementation and synthetic replicas. No independent other-match dataset was available. Automatic tags can be wrong, tracking jumps can resemble real kicks, and neither referee whistles nor ball height are observed. Heuristic dead-ball recall may be low. Long occlusions can conceal an ownership change; Viterbi can overmerge weak short possessions. Team aggregates can be distorted by duplicate tracks.

Linux execution is supported by portable code and a Windows/Linux CI workflow, but only Windows execution was verified during local delivery. Exact cross-platform byte reproducibility remains to be checked on real data; the verified repeated-run claim applies to this Windows environment.

<!-- pagebreak -->

## Reproduction, compliance and handoff

### 12. Reproduce the exact pipeline

Use Python 3.10 or 3.11. Install requirements-goalgorithm.txt in an isolated environment. The tested runtime is Python 3.10.11, NumPy 2.2.6 and pandas 2.3.3, with all runtime dependencies pinned. No GPU is needed. A few hundred MiB were sufficient for synthetic inference; allow at least 1 GiB free memory for practical headroom.

Put the original authorized files under data/tactical/: BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt, and BAR-ATH_tactical_smart_tagging.xml. Then execute locally:

```text
python run_goalgorithm.py --data-dir ./data --output Goalgorithm.csv
```

This single command performs inference, writes exactly frame,label and 199,319 rows, and gates success on the unchanged official validator's exit status. The optional --config selects a committed candidate; omitted configuration equals configs/temporal.json. Repeat runs use deterministic computation and fixed CSV newlines. The local diagnostics sidecar includes a SHA-256 digest and the complete resolved configuration.

Validate the final file before upload:

```text
python -X utf8 evaluation/validate_submission.py Goalgorithm.csv
```

To reproduce the synthetic engineering checks without match assets:

```text
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.compare_synthetic
python -X utf8 -m tests.smoke_full --slots 46
```

### 13. Data-use compliance

Development used public repository code/documentation and invented synthetic fixtures. The licensed archive present outside the checkout remained unopened. No restricted records, video, tracking extracts or derived match outputs were sent to hosted services. No hand labelling, private ground truth, third-party human annotations or competition-label training was used.

The standalone inference runner has no network calls and does not print tracking content. Run it in a local terminal outside hosted assistant inspection. Keep its CSV and diagnostics private except for the authorized competition submission. Git ignore rules exclude assets, derived arrays, generated CSVs and local diagnostics. The PDF contains only implementation details, published reference values and synthetic results.

The official evaluation, starter code and match constants remain unmodified. The baseline fallback is attributable to the MIT-licensed official repository. See README.md, DATA.md, RULES.md, EVALUATION.md and SUBMISSION.md in that repository for authoritative competition definitions.

### 14. Remaining participant actions

Generate the actual CSV from the authorized local bundle and require validator exit status zero. Confirm the public repository contains this implementation and supply its URL. Upload the actual CSV and this technical document through the form emailed to registered participants before November 17, 2026. The technical document is complete; match execution and the human submission action remain outstanding.

No synthetic CSV may be submitted as Goalgorithm's match prediction. No claim of beating spellF1 0.2433 is justified before a permitted independent evaluation or the organizer's private evaluation.
