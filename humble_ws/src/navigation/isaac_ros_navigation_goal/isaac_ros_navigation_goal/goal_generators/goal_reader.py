# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .goal_generator import GoalGenerator
from pathlib import Path

import yaml


class GoalReader(GoalGenerator):
    def __init__(self, file_path):
        self.__file_path = file_path
        self.__generator = self.__get_goal()

    def generate_goal(self, max_num_of_trials=1000):
        try:
            return next(self.__generator)
        except StopIteration:
            return

    def __get_goal(self):
        path = Path(self.__file_path)
        suffix = path.suffix.lower()
        if suffix not in (".yaml", ".yml"):
            raise ValueError(
                f"Unsupported experience file '{path}'. Only .yaml/.yml files are supported."
            )

        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Experience file must contain a YAML mapping: {path}")
        goals = config.get("goals")
        if not isinstance(goals, list) or len(goals) == 0:
            raise ValueError(f"Experience file must include a non-empty 'goals' list: {path}")
        for idx, goal in enumerate(goals):
            if not isinstance(goal, list) or len(goal) != 7:
                raise ValueError(
                    f"Goal at index {idx} must contain exactly 7 values [x, y, z, qw, qx, qy, qz]"
                )
            yield [float(v) for v in goal]
