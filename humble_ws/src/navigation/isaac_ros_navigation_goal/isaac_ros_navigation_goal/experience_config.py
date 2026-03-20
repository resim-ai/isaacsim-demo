#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from ament_index_python.packages import get_package_share_directory

TEST_CONFIG_PATH = Path("/tmp/resim/test_config.json")


def _package_assets_dir() -> Path:
    return Path(get_package_share_directory("isaac_ros_navigation_goal")) / "assets"


def default_experience_yaml_path() -> Path:
    return _package_assets_dir() / "experience.yaml"


def resolve_experience_location() -> Path:
    if TEST_CONFIG_PATH.exists():
        with TEST_CONFIG_PATH.open("r", encoding="utf-8") as test_config_file:
            test_config = json.load(test_config_file)
        return Path(test_config["experienceLocation"])

    return default_experience_yaml_path()


def _validate_initial_pose(initial_pose: Any) -> List[float]:
    if not isinstance(initial_pose, list):
        raise ValueError("experience initial_pose must be a list of 7 numbers")
    if len(initial_pose) != 7:
        raise ValueError("experience initial_pose must contain exactly 7 values [x, y, z, qw, qx, qy, qz]")
    return [float(v) for v in initial_pose]


def _coerce_goals_from_yaml(goals: Any) -> List[List[float]]:
    if not isinstance(goals, list) or len(goals) == 0:
        raise ValueError("experience goals must be a non-empty list of goal arrays")

    validated_goals: List[List[float]] = []
    for idx, goal in enumerate(goals):
        if not isinstance(goal, list) or len(goal) != 7:
            raise ValueError(
                f"experience goal at index {idx} must contain exactly 7 values [x, y, z, qw, qx, qy, qz]"
            )
        try:
            validated_goals.append([float(v) for v in goal])
        except (TypeError, ValueError) as ex:
            raise ValueError(f"experience goal at index {idx} contains non-numeric values") from ex
    return validated_goals


def _coerce_optional_string_field(config: Dict[str, Any], key: str) -> Optional[str]:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"experience {key} must be a string")
    return value


def load_experience_config(
    *,
    experience_location: Optional[Union[str, Path]] = None,
    default_initial_pose: Optional[List[float]] = None,
    default_isaacsim_entity: str = "",
) -> Dict[str, Any]:
    """Resolve experience config from an explicit path, /tmp/resim/test_config.json, or package defaults."""
    resolved_experience_location = (
        Path(experience_location) if experience_location is not None else resolve_experience_location()
    )
    if not resolved_experience_location.exists():
        raise FileNotFoundError(f"Experience file not found: {resolved_experience_location}")
    suffix = resolved_experience_location.suffix.lower()

    if suffix not in (".yaml", ".yml"):
        raise ValueError(
            f"Unsupported experience file '{resolved_experience_location}'. "
            "Only .yaml/.yml experience files are supported."
        )

    with resolved_experience_location.open("r", encoding="utf-8") as experience_file:
        config = yaml.safe_load(experience_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experience file must contain a YAML mapping: {resolved_experience_location}")

    if "initial_pose" not in config and default_initial_pose is None:
        raise ValueError(f"Missing 'initial_pose' in experience file: {resolved_experience_location}")

    goals = _coerce_goals_from_yaml(config.get("goals"))
    goal_count = len(goals)
    initial_pose = _validate_initial_pose(config.get("initial_pose", default_initial_pose))
    isaacsim_entity = str(config.get("isaacsim_entity", default_isaacsim_entity))
    world_uri = str(config.get("world_uri", ""))
    namespace = _coerce_optional_string_field(config, "namespace")
    nav2_params_file = _coerce_optional_string_field(config, "nav2_params_file")
    map_file = _coerce_optional_string_field(config, "map_file")
    rviz_config_file = _coerce_optional_string_field(config, "rviz_config_file")

    return {
        "experience_path": str(resolved_experience_location),
        "goal_count": goal_count,
        "initial_pose": initial_pose,
        "isaacsim_entity": isaacsim_entity,
        "world_uri": world_uri,
        "namespace": namespace,
        "nav2_params_file": nav2_params_file,
        "map_file": map_file,
        "rviz_config_file": rviz_config_file,
    }
