# Goalgorithm repository audit

## Starting state

The workspace contained a competition dataset ZIP and no checkout. A fresh clone of the requested team repository was created in `more-than-a-hack-2027/`. Its `main` branch was clean at `f8f2629ff1e0985f81d7e59d8413678514ee1530`, also the official upstream HEAD. No pre-existing team implementation or uncommitted changes were found in this workspace. Other filesystem locations were not searched for team work.

## Operational components

The official NumPy baseline, pandas/NumPy scorer and validator are operational. Public baseline metrics were reproduced for pipeline verification only. The scorer needs `python -X utf8` or `PYTHONUTF8=1` in this Windows console to print the non-ASCII workspace path. No scoring definitions require changes. Official full-match smooth50 spellF1 **0.2433** is a published reference, not a Goalgorithm result.

## Critical assumptions and weaknesses

- Tactical ATD is numbered 1..199319; submission indices are 0..199318. The distinct IdFrame offsets in `games.py` belong to the ground-truth counter, not delivered ATD.
- Graded halves are inclusive 11185..90651 and 110808..190197. Second-half kickoff is not the start of the second graded block.
- Tracking coordinates must scale by 105.3 x 68.0 metres; metadata rounds length to 105.
- Track slots are tracklets, not persistent people. Team metadata must reject unknown team references instead of silently assigning away.
- Starter raw alignment checking only compares endpoints and count; compensating duplicates/gaps can pass. The new reader checks every consecutive ID and exact slot count.
- Starter missing-ball classification is D; this confuses occlusion and stoppage. Majority filtering is centred and counts H/A votes, rather than voting equally among three states.
- Smart Tagging intervals are tactical-video seconds; team names include `Atletic Club` and `N/A`. Defensive tags imply the opposite team only as weak evidence. Set-piece windows do not identify the exact restart instant.
- Homography maps normalized pitch to pixels. Its loader does not itself enforce ordered frame indices; the companion check does. Homography/video are unnecessary for a pitch-coordinate implementation and are not read.
- The official validator tolerates extra columns despite the documented two-column contract. The new writer produces exactly `frame,label` and also checks the exact header.

## Feasible delivery and data boundary

A CPU-only deterministic inference pipeline, official-baseline fallback, four model configurations, synthetic scenario tests, full-length synthetic integration, validation gating and technical documentation can be completed without licensed data access. No independent labelled other-match dataset is available here, so real performance improvements cannot be established.

The nearby licensed archive has not been opened or inspected. Real inference and real CSV validation must be performed by the participant in a standalone local terminal, outside hosted assistant inspection. Required inputs: tactical ATD metadata and raw tracking; Smart Tagging for the default fusion method. No video, broadcast alignment or GPU is needed.

## Submission requirements

A validated 199319-row team CSV generated locally from the authorized bundle; a public repository with pinned dependencies and reproduction command; a technical document; final upload through the organizers' registration email form before November 17, 2026. A synthetic output is never a valid competitive entry.
