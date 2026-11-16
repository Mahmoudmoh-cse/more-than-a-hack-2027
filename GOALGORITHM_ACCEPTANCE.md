# Goalgorithm final acceptance record

Date: 2026-11-16. Scope: public source code and synthetic data only.

| Gate | Status | Evidence |
| --- | --- | --- |
| Source files compile | PASS | Python compileall on goalgorithm/, tests/, scripts/ and runner; report generator executed successfully. |
| Dependencies valid | PASS | Clean .venv installation of requirements-goalgorithm.txt; pip check: no broken requirements. |
| Imports work | PASS | Suite imports every inference module and official helpers; full CLI starts in real subprocesses. |
| Synthetic tracking tests | PASS | 30/30 unittest cases, including strict parser, team mapping and metric scaling. |
| Frame alignment | PASS | ATD one-based round trips, duplicate/gap rejection, official graded endpoints, tag seconds and half-open intervals. |
| Temporal state tests | PASS | Controlled switches, brief deflections, short valid possessions, occlusions and exhaustive Viterbi optimum. |
| Dead-ball and restart tests | PASS | Out-of-bounds D, tagged stationary D, live moving restart; stationary live ball and missing ball do not automatically become D. |
| Deterministic inference | PASS | Repeated outputs for all four candidates; two full 46-slot CLI outputs have identical SHA-256. |
| Full-length synthetic CLI | PASS | 199319 rows; validator exit 0; 40462 ungraded X; both halves processed independently. |
| Full-match licensed inference | BLOCKED | Archive not opened in hosted development; participant must execute standalone locally. |
| Real submission validation | BLOCKED | No real Goalgorithm.csv has been generated; synthetic validation does not satisfy this gate. |
| Official scorer unmodified | PASS | No Git diff to evaluation/, starter/ or games.py. Official baseline public-window installation check and synthetic scorer execution exit 0. |
| Baseline fallback | PASS | Calls unchanged official nearest_player_labels and smooth; synthetic equivalence test passed. Full real-data regeneration awaits local execution. |
| Technical documentation | PASS | README, audit, progress, technical Markdown, config guide, acceptance record and five-page PDF; all pages rendered and visually reviewed. |
| Reproduction command | PASS | Exact runner command exercised twice with a full-sized fabricated bundle; pinned environment suite passed. Real-data reproduction remains blocked above. |
| Restricted data excluded from Git | PASS | Archive outside checkout unopened; ignored asset/CSV/array/diagnostic paths; staged-path gate and staged-content review. Only public starter assets are inherited. |
| No manual match annotations | PASS | No competition data inspected/annotated; generated fixtures are invented scenes only. |
| No training/tuning on public GT | PASS | No training code, GT imports or GT reads in inference. Only unchanged official-baseline installation verification used the public window. |
| Git changes reviewed | PASS | Explicit staging, data-leak path checks, git diff --check and unchanged official-source check. Meaningful audit/implementation commits preserved. |
| Linux runtime verification | BLOCKED | Portable code and CI workflow supplied; local execution host is Windows. No Linux run claimed. |
| Actual performance gain over 0.2433 | BLOCKED | No independent real-match validation; organizer private-match spellF1 remains unknown. |
| Final competition upload | BLOCKED | Participant must generate/validate the real CSV and submit the three deliverables through the emailed form. |

## Actual execution results

Existing interpreter: 30 tests passed in 15.551 s while other jobs ran. Clean pinned virtual environment: 30 tests passed in 13.603 s. Windows 64-bit, Python 3.10.11, NumPy 2.2.6, pandas 2.3.3.

Full synthetic runs: 26.3838 and 25.4080 s including ingestion and official validation; 285.2734 and 285.1094 MiB process peak memory (not tracemalloc and not total OS memory). Two-run fixture generation/execution total: 56.7660 s. This measures synthetic throughput, not a licensed-match runtime.

Identical synthetic CSV SHA-256: `086763f0c2f77c61a96f279110e1b37babc5044bad8d09a5ac3afc7a3eb30967`.

Full synthetic output: 320 possessions, mean 19.857125 s, 0% graded X, 20.300122% overall X, approximately 2.40% missing-ball observations per half. These describe fabricated inputs only. Real match possession statistics are unknown.

On the separate 2400-frame handcrafted fixture: baseline macroF1/spellF1 0.5591/0.6000; geometry 0.5390/0.3273; tagging 0.6354/0.6154; temporal 1.0000/1.0000. The fixture is not held-out validation or evidence of competitive strength.

Official baseline smooth50 public-window installation check: macroF1 0.6207, spellF1 0.3256. Organizer-published full-match baseline spellF1 0.2433. Neither is a Goalgorithm score. The first official-scorer invocation reproduced scores but exited on a Windows cp1252 path-printing error; re-running unchanged code with `-X utf8` completed successfully. No model selection depended on those scores.

## Reproduce acceptance

```text
python -m pip install -r requirements-goalgorithm.txt
python -m pip check
python -X utf8 -m compileall -q goalgorithm tests scripts run_goalgorithm.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.compare_synthetic
python -X utf8 -m tests.smoke_full --slots 46
```

Only the participant's local run can close the real inference/validation gates. The code's submission writer runs the official validator and fails on nonzero status. No blocked gate is being treated as a pass.
