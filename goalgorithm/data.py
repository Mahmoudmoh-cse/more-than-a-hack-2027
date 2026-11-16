"""Strict readers. Error messages never echo restricted records or metadata."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, asset_paths
from starter.load_smart_tagging import load as load_tagging


@dataclass
class Tracking:
    players: np.ndarray  # frame, slot, xy; metres
    ball: np.ndarray  # frame, xy; metres
    teams: np.ndarray  # slot -> 0 home / 1 away
    team_names: tuple = ("Barcelona", "Atletic Club")
    invalid_coordinates: int = 0


def read_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    ids = [team.get("id") for team in teams]
    if len(ids) != 2 or None in ids or len(set(ids)) != 2:
        raise ValueError("ATD metadata must define exactly two distinct teams")
    members = [player.get("teamId") for player in root.iter("Player")]
    if not members or any(team not in ids for team in members):
        raise ValueError("ATD metadata has missing or unknown slot team membership")
    mapping = np.asarray([ids.index(team) for team in members], dtype=np.int8)
    if set(mapping.tolist()) != {0, 1}:
        raise ValueError("ATD metadata must include slots for both teams")
    names = tuple((team.findtext("Name") or "").strip() for team in teams)
    # FIFA/EPTS first-team home convention; reject an explicitly reversed known pair.
    if "barcelona" in names[1].casefold() or "athletic" in names[0].casefold() or "atletic" in names[0].casefold():
        raise ValueError("ATD home/away names disagree with the documented match")
    return mapping, names


def _xy(group):
    fields = group.strip().split(",")
    if len(fields) < 3:
        raise ValueError("ATD coordinate group requires x,y,speed")
    # Parse speed for format checking; derive motion from positions to gate slot jumps.
    values = [float(value) for value in fields[:3]]
    return values[:2]


def read_tracking(meta_path, raw_path, n_frames, pitch):
    teams, names = read_metadata(meta_path)
    coordinates = np.full((n_frames, len(teams) + 1, 2), np.nan, dtype=np.float32)
    count = 0
    with open(raw_path, encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if index >= n_frames:
                raise ValueError("ATD record count exceeds the tactical video length")
            fields = line.strip().split(":")
            if len(fields) != 3 or int(fields[0]) != index + 1:
                raise ValueError("ATD IDs must be exactly consecutive from one")
            groups = fields[1].split(";")
            if len(groups) != len(teams):
                raise ValueError("ATD slot count does not match metadata")
            coordinates[index] = [_xy(group) for group in groups] + [_xy(fields[2])]
            count += 1
    if count != n_frames:
        raise ValueError("ATD record count differs from the tactical video length")
    # Small excursions beyond the pitch are meaningful dead-ball evidence. Gross values
    # and partial/infinite coordinates are unusable; do not clip them onto the pitch.
    finite = np.isfinite(coordinates).all(axis=2)
    plausible = ((coordinates >= -0.1) & (coordinates <= 1.1)).all(axis=2)
    invalid = (~np.isnan(coordinates).all(axis=2)) & ~(finite & plausible)
    coordinates[~(finite & plausible)] = np.nan
    coordinates *= np.asarray(pitch, dtype=np.float32)
    return Tracking(coordinates[:, :-1], coordinates[:, -1], teams, names, int(invalid.sum()))


def load_inputs(data_dir, need_tags=True):
    paths = asset_paths(data_dir=data_dir)
    from pathlib import Path
    needed = ["atd_meta", "atd_raw"] + (["smart_tagging"] if need_tags else [])
    if any(not Path(paths[name]).is_file() for name in needed):
        raise ValueError("Required tactical assets missing; see GOALGORITHM_README.md")
    game = GAMES[GAME]
    tracking = read_tracking(paths["atd_meta"], paths["atd_raw"], game["n_video_frames"], game["pitch"])
    events = load_tagging(data_dir=data_dir) if need_tags else []
    return tracking, events
