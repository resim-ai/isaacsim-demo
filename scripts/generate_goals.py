import math
from pathlib import Path
import random
from itertools import permutations

GOAL_LOCATIONS = [
    (-6.0, -5.0),
    (-6.0, -5.0),
    (-6.0, -10.0),
    (7.0, -10.0),
    (-6.0, 10.0),
    (-9.0, 14.0),
    (7.0, 7.0),
    (7.0, 14.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.4, 4.5),
    (-7.4, -9.4),
    (5.7, -8.9),
    (6.62, 12.93),
    (1.77, -0.64),
    (6.63, 0.56),
    (5.94, -3.39),
]

all_permutations = permutations(GOAL_LOCATIONS, 3)
num_experiences = 50
goal_lists = random.sample(list(all_permutations), num_experiences)


output_dir = Path("goals")
output_dir.mkdir(exist_ok=True)
for i, goal_list in enumerate(goal_lists):
    goal_list_path = output_dir / f"goals_{i + 1}.txt"
    with open(goal_list_path, "w", encoding="utf-8") as fp:
        for j, goal in enumerate(goal_list):
            theta = random.uniform(0.0, math.tau)
            goal = [*goal, 0.0, 0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)]
            maybe_newline = "\n" if j < len(goal_list) - 1 else ""
            fp.write(" ".join(map(str, goal)) + maybe_newline)
