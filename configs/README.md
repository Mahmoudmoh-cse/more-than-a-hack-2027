# Fixed candidates

## Phase 1 (v1) preserved baselines & ablations

`baseline.json`: official raw reader plus +/-50-frame H/A vote, reproducing the published method. It deliberately retains the baseline's missing-ball weakness. The command-line branch uses the unmodified official functions and smooths the complete recording for exact reproduction.

`geometry.json`: reliable geometric evidence with decaying controller memory, no tactical prior or D inference. This is an ablation, not a complete stoppage model.

`tagging.json`: geometry plus bounded offensive/defensive Smart Tagging priors, no temporal chain or D inference.

`temporal.json`: v1 primary configuration, all v1 features plus explicit v1 stoppage evidence and half-local Viterbi. Retained unmodified as the safe v1 baseline (`Config()` defaults match this file).

## Phase 2 (v2) modular improvements & ablations

`v2_stoppage.json`: v1 control + v1 Viterbi combined with `goalgorithm/quality.py` and `goalgorithm/stoppage.py` (boundary-exit-to-gap detection, dropout-bridged stillness, set-piece `Type`/`GOALS` fusion, restart-kick snapping, pre-kickoff dead intervals, and live-vs-dead-vs-uncertain separation).

`v2_control.json`: `goalgorithm/quality.py` + `goalgorithm/control.py` (centered kinematics, same-team slot deduplication, slot-agnostic continuity, bystander pass-flight attenuation, duel damping, same-team pass continuity bridge, and acquisition edges) with v1 stoppage + v1 Viterbi.

`v2_decoder.json`: v1 features + v1 stoppage with `goalgorithm/segment_decoder.py` (exact event-modulated 3-state DP).

`v2.json`: complete candidate v2 combining `quality.py`, `stoppage.py`, `control.py`, and `segment_decoder.py`. New parameters: `dead_enter_cost=4.5`, `dead_exit_cost=4.5`, `memory_weight=0.55`, `pass_bridge_weight=0.65`, `restart_prior_weight=0.35`.

All distances are metres and times seconds except costs/weights, `baseline_window` (frames), and speed/acceleration features (SI units). No parameter was fitted to BAR-ATH competition labels or the public five-minute ground truth. Independent validation uses synthetic adversarial/metamorphic fixtures and public external matches from `metrica-sports/sample-data` (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`).
