"""Versioned engineering parameters. No fitting to the competition ground truth."""

from dataclasses import asdict, dataclass
import json
import math


SUPPORTED_MODELS = {
    "baseline",
    "geometry",
    "tagging",
    "temporal",
    "v2_stoppage",
    "v2_control",
    "v2_decoder",
    "v2",
}
TAG_MODELS = {"tagging", "temporal", "v2_stoppage", "v2_control", "v2_decoder", "v2"}


@dataclass(frozen=True)
class Config:
    model: str = "temporal"
    baseline_window: int = 50
    max_ball_speed: float = 45.0
    max_player_speed: float = 13.0
    position_slack: float = 0.6
    interpolation_seconds: float = 0.32
    velocity_seconds: float = 0.20
    control_radius: float = 2.5
    density_radius: float = 5.0
    geometry_weight: float = 1.8
    tag_weight: float = 0.45
    memory_seconds: float = 6.0
    switch_cost: float = 7.0
    dead_switch_cost: float = 5.0
    stationary_speed: float = 0.65
    stationary_seconds: float = 1.5
    restart_lookback_seconds: float = 8.0
    outside_margin: float = 0.6
    outside_seconds: float = 0.5
    # v2-specific parameters (unused by v1 models; defaults preserved for backward compatibility)
    dead_enter_cost: float = 4.5
    dead_exit_cost: float = 4.5
    memory_weight: float = 0.55
    pass_bridge_weight: float = 0.65
    restart_prior_weight: float = 0.35

    def validate(self):
        if self.model not in SUPPORTED_MODELS:
            raise ValueError("unsupported model in configuration")
        if type(self.baseline_window) is not int or self.baseline_window < 0:
            raise ValueError("baseline_window must be a nonnegative integer")
        for key, value in asdict(self).items():
            if key not in {"model", "baseline_window"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("configuration values must be finite positive numbers")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("configuration values must be finite positive numbers")
        return self


def load_config(path=None):
    if path is None:
        return Config().validate()
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    return Config(**values).validate()
