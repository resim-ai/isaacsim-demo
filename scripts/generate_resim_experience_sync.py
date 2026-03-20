#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_SYSTEMS = ("Isaac Sim",)
DEFAULT_CONTAINER_TIMEOUT_SECONDS = 1800


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(
        description="Generate resim_experience_sync.yaml from nav2 experience files."
    ).parse_args(
        namespace=argparse.Namespace(
            experiences_dir=repo_root / "builds/nav2/experiences",
            output=repo_root / "resim_experience_sync.yaml",
        )
    )


class SyncDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def normalize_custom_fields(raw_custom_fields: list[dict] | None) -> list[dict]:
    normalized_fields = []

    for field in raw_custom_fields or []:
        if field.get("name") == "Building":
            continue

        values = field.get("values", field.get("value", []))
        normalized_fields.append(
            {
                "name": field["name"],
                "type": field.get("type", "text"),
                "values": list(values),
            }
        )

    return normalized_fields


def parse_experience(path: Path, experiences_dir: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.BaseLoader)

    experience = {
        "name": data["name"],
        "description": data["description"],
        "systems": list(DEFAULT_SYSTEMS),
        "containerTimeoutSeconds": DEFAULT_CONTAINER_TIMEOUT_SECONDS,
        "locations": [f"/experiences/{path.relative_to(experiences_dir).as_posix()}"],
    }

    custom_fields = normalize_custom_fields(data.get("customFields"))
    if custom_fields:
        experience["customFields"] = custom_fields

    return experience


def read_managed_test_suites(output_path: Path) -> list[dict] | None:
    if not output_path.exists():
        return None

    with output_path.open(encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.BaseLoader) or {}

    return data.get("managedTestSuites")


def generate_sync_file(experiences_dir: Path, output_path: Path) -> None:
    experiences = [
        parse_experience(path, experiences_dir)
        for path in sorted(experiences_dir.rglob("*.yaml"))
    ]
    experiences.sort(key=lambda experience: experience["name"])

    document = {"experiences": experiences}

    managed_test_suites = read_managed_test_suites(output_path)
    if managed_test_suites:
        document["managedTestSuites"] = managed_test_suites

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            document,
            handle,
            Dumper=SyncDumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
            width=1000,
        )


def main() -> None:
    args = parse_args()
    generate_sync_file(args.experiences_dir, args.output)


if __name__ == "__main__":
    main()
