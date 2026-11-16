# Fixed candidates

`baseline.json`: official raw reader plus +/-50-frame H/A vote, reproducing the published method. It deliberately retains the baseline's missing-ball weakness. The command-line branch uses the unmodified official functions and smooths the complete recording for exact reproduction.

`geometry.json`: reliable geometric evidence with decaying controller memory, no tactical prior or D inference. This is an ablation, not a complete stoppage model.

`tagging.json`: geometry plus bounded offensive/defensive Smart Tagging priors, no temporal chain or D inference.

`temporal.json`: primary, all features plus explicit stoppage evidence and half-local Viterbi. This file lists every parameter. Other configurations inherit numerical defaults from the versioned `Config` dataclass. An absent `--config` uses exactly the primary defaults.

All distances are metres and times seconds except costs/weights, `baseline_window` (frames), and speed/acceleration features (SI units). 45 m/s bounds interpolation; 13 m/s gates player continuity; 0.6 m slack tolerates jitter. Interpolation spans at most 0.32 s and receives only 0.35 initial reliability. A 0.2 s velocity window reduces jitter. Near control uses a 2.5 m scale; density uses 5 m. Controller memory decays over 6 s. Viterbi switch costs 7 (team) / 5 (D) require accumulated evidence; strong recovery reduces team cost by at most 25%. Stoppage requires observed outside-pitch persistence or stillness supported by a set-piece window or collective quiet near a boundary. No parameter was fitted to competition labels.

These are engineering candidates, not an empirically established ranking. Do not select between them using the public five-minute ground truth. Synthetic checks establish behavior, not match performance. Use independently licensed other-match validation before making performance claims.
