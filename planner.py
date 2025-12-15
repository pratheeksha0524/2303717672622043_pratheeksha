

import json
import math
import sys


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_centroid(corners):
    """
    Area-weighted centroid (shoelace formula)
    Guaranteed to lie inside a convex polygon
    """
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(corners)

    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    area *= 0.5
    cx /= (6 * area)
    cy /= (6 * area)

    return (cx, cy)


# --------------------------------------------------
# Unified Solver (Milestone 1 & 2)
# --------------------------------------------------

def solve(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    stage_velocity = data["StageVelocity"]
    start = tuple(data["InitialPosition"])

    # IMPORTANT:
    # - Keep die order EXACTLY as input
    # - Use polygon centroid (not corner average)
    visit_points = [
        polygon_centroid(d["Corners"])
        for d in data["Dies"]
    ]

    path = [start]
    total_time = 0.0
    current = start

    for point in visit_points:
        total_time += distance(current, point) / stage_velocity
        path.append(point)
        current = point

    output = {
        "TotalTime": total_time,
        "Path": path
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    DEFAULT_INPUT = "Input_Milestone2_Testcase3.json"
    DEFAULT_OUTPUT = "TestCase_2_3.json"

    if len(sys.argv) == 3:
        solve(sys.argv[1], sys.argv[2])
    else:
        solve(DEFAULT_INPUT, DEFAULT_OUTPUT)

    print("Output generated successfully.")
