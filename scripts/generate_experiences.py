#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYNC_CONFIG_NAME = "resim_experience_sync.yaml"
TEST_SUITE_NAME = "Hospital Demo"
SMOKE_TEST_SUITE_NAME = "Hospital Demo Smoke"


ROUTES = (
    {
        "slug": "south_hallway_from_room",
        "title": "South Hallway",
        "direction_name": "From Room",
        "direction_field": "From Patient Room",
        "from_desc": "patient room",
        "to_desc": "lobby",
        "initial_pose": "[24.0, 24.0, 0.0, 0.70711, 0.0, 0.0, 0.70711]",
        "goal": "[11.5, 6.0, 0.0, 0.0, 0.0, 0.0, -1.0]",
    },
    {
        "slug": "south_hallway_to_room",
        "title": "South Hallway",
        "direction_name": "To Room",
        "direction_field": "To Patient Room",
        "from_desc": "lobby",
        "to_desc": "patient room",
        "initial_pose": "[11.5, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
        "goal": "[24.0, 24.0, 0.0, -0.70711, 0.0, 0.0, 0.70711]",
    },
    {
        "slug": "east_hallway_from_north_hallway",
        "title": "East Hallway",
        "direction_name": "From North Hallway",
        "direction_field": "From North Hallway",
        "from_desc": "north hallway",
        "to_desc": "lobby",
        "initial_pose": "[-34.0, 13.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
        "goal": "[11.5, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
    },
    {
        "slug": "east_hallway_to_north_hallway",
        "title": "East Hallway",
        "direction_name": "To North Hallway",
        "direction_field": "To North Hallway",
        "from_desc": "lobby",
        "to_desc": "north hallway",
        "initial_pose": "[11.5, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
        "goal": "[-34.0, 13.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
    },
    {
        "slug": "lobby_from_north_hallway",
        "title": "Lobby",
        "direction_name": "From North Hallway",
        "direction_field": "From North Hallway",
        "from_desc": "north hallway",
        "to_desc": "lobby",
        "initial_pose": "[-23.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
        "goal": "[11.5, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0]",
    },
    {
        "slug": "lobby_to_north_hallway",
        "title": "Lobby",
        "direction_name": "To North Hallway",
        "direction_field": "To North Hallway",
        "from_desc": "lobby",
        "to_desc": "north hallway",
        "initial_pose": "[11.5, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
        "goal": "[-23.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
    },
)

VARIANTS = (
    {
        "slug": "bright_ped",
        "lighting_title": "Bright",
        "lighting_desc": "bright",
        "ped_title": "with Pedestrians",
        "ped_desc": "with pedestrians",
        "ped_value": "Yes",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_ped_bright.usda",
    },
    {
        "slug": "bright_no_ped",
        "lighting_title": "Bright",
        "lighting_desc": "bright",
        "ped_title": "without Pedestrians",
        "ped_desc": "without pedestrians",
        "ped_value": "No",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_no_ped_bright.usda",
    },
    {
        "slug": "dim_ped",
        "lighting_title": "Dim",
        "lighting_desc": "dim",
        "ped_title": "with Pedestrians",
        "ped_desc": "with pedestrians",
        "ped_value": "Yes",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_ped_mid.usda",
    },
    {
        "slug": "dim_no_ped",
        "lighting_title": "Dim",
        "lighting_desc": "dim",
        "ped_title": "without Pedestrians",
        "ped_desc": "without pedestrians",
        "ped_value": "No",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_no_ped_mid.usda",
    },
    {
        "slug": "dark_ped",
        "lighting_title": "Dark",
        "lighting_desc": "dark",
        "ped_title": "with Pedestrians",
        "ped_desc": "with pedestrians",
        "ped_value": "Yes",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_ped_dark.usda",
    },
    {
        "slug": "dark_no_ped",
        "lighting_title": "Dark",
        "lighting_desc": "dark",
        "ped_title": "without Pedestrians",
        "ped_desc": "without pedestrians",
        "ped_value": "No",
        "world_uri": "/tmp/resim/assets/collected_hospital_demo/hospital_demo_one_robot_no_ped_dark.usda",
    },
)

TEMPLATE = """name: Hospital {title} Navigation {direction_name} - {lighting_title} {ped_title}
description: Robot navigates from {from_desc} to {to_desc} in {lighting_desc} environment {ped_desc}.
initial_pose: {initial_pose}
isaacsim_entity: "/World/Nova_Carter_ROS_1"
goals:
  - {goal}
world_uri: "{world_uri}"
customFields:
- name: Lighting
  type: text
  values:
  - {lighting_title}
- name: Pedestrians
  type: text
  value:
  - {ped_value}
- name: Area
  type: text
  values:
  - {title}
- name: Direction
  type: text
  value:
  - {direction_field}
"""


def build_template_values(
    route: dict[str, str], variant: dict[str, str]
) -> dict[str, str]:
    template_values = {
        key: value for key, value in route.items() if key != "slug"
    }
    template_values.update(
        {key: value for key, value in variant.items() if key != "slug"}
    )
    return template_values


def build_scenario(
    route: dict[str, str], variant: dict[str, str]
) -> dict[str, Any]:
    template_values = build_template_values(route, variant)
    scenario: dict[str, Any] = dict(template_values)
    scenario["filename"] = f"{route['slug']}_{variant['slug']}.yaml"
    scenario["name"] = (
        f"Hospital {template_values['title']} Navigation "
        f"{template_values['direction_name']} - "
        f"{template_values['lighting_title']} {template_values['ped_title']}"
    )
    scenario["description"] = (
        f"Robot navigates from {template_values['from_desc']} "
        f"to {template_values['to_desc']} in "
        f"{template_values['lighting_desc']} environment "
        f"{template_values['ped_desc']}."
    )
    scenario["custom_fields"] = [
        {
            "name": "Lighting",
            "type": "text",
            "values": [template_values["lighting_title"]],
        },
        {
            "name": "Pedestrians",
            "type": "text",
            "values": [template_values["ped_value"]],
        },
        {
            "name": "Area",
            "type": "text",
            "values": [template_values["title"]],
        },
        {
            "name": "Direction",
            "type": "text",
            "values": [template_values["direction_field"]],
        },
    ]
    scenario["yaml"] = TEMPLATE.format(**template_values)
    return scenario


def get_scenarios() -> list[dict[str, Any]]:
    return [
        build_scenario(route, variant)
        for route in ROUTES
        for variant in VARIANTS
    ]


def write_scenarios(
    output_dir: Path, scenarios: list[dict[str, Any]]
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scenario in scenarios:
        path = output_dir / scenario["filename"]
        path.write_text(scenario["yaml"], encoding="ascii")
        written.append(path)
    return written


def build_sync_config(scenarios: list[dict[str, Any]]) -> str:
    smoke_scenarios = [
        scenario for scenario in scenarios if scenario["filename"].endswith("bright_no_ped.yaml")
    ]
    lines = ["experiences:"]
    for scenario in scenarios:
        location = f"/experiences/{scenario['filename']}"
        lines.extend(
            (
                f"  - name: {json.dumps(scenario['name'])}",
                f"    description: {json.dumps(scenario['description'])}",
                "    systems: [\"Isaac Sim\"]",
                "    containerTimeoutSeconds: 1800",
                "    locations:",
                f"      - {json.dumps(location)}",
                "    customFields:",
            )
        )
        for custom_field in scenario["custom_fields"]:
            lines.extend(
                (
                    f"      - name: {json.dumps(custom_field['name'])}",
                    f"        type: {json.dumps(custom_field['type'])}",
                    "        values:",
                )
            )
            for value in custom_field["values"]:
                lines.append(f"          - {json.dumps(value)}")

    lines.extend(
        (
            "managedTestSuites:",
            f"  - name: {json.dumps(TEST_SUITE_NAME)}",
            "    experiences:",
        )
    )
    for scenario in scenarios:
        lines.append(f"      - {json.dumps(scenario['name'])}")
    lines.extend(
        (
            f"  - name: {json.dumps(SMOKE_TEST_SUITE_NAME)}",
            "    experiences:",
        )
    )
    for scenario in smoke_scenarios:
        lines.append(f"      - {json.dumps(scenario['name'])}")

    return "\n".join(lines) + "\n"


def write_sync_config(
    sync_config_path: Path, scenarios: list[dict[str, Any]]
) -> Path:
    sync_config_path.parent.mkdir(parents=True, exist_ok=True)
    sync_config_path.write_text(build_sync_config(scenarios), encoding="ascii")
    return sync_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 36 hospital nav2 experience scenarios."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "builds" / "nav2" / "experiences",
        help="Directory where the scenario YAML files will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the file paths that would be written without modifying files.",
    )
    parser.add_argument(
        "--sync-config-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / SYNC_CONFIG_NAME,
        help=f"Path for the generated ReSim experience syncing config. Defaults to {SYNC_CONFIG_NAME}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    sync_config_path = (
        args.sync_config_path.resolve()
        if args.sync_config_path
        else output_dir / SYNC_CONFIG_NAME
    )
    scenarios = get_scenarios()

    if args.dry_run:
        for scenario in scenarios:
            print(output_dir / scenario["filename"])
        print(f"total: {len(scenarios)}")
        print(f"sync config: {sync_config_path}")
        return 0

    written = write_scenarios(output_dir, scenarios)
    write_sync_config(sync_config_path, scenarios)
    print(f"wrote {len(written)} scenarios to {output_dir}")
    print(f"wrote ReSim sync config to {sync_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
