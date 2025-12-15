


import json
import math
import os
from typing import List, Tuple, Dict, Any
import numpy as np


class DiePathOptimizer:

    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def calculate_die_center(corners: List[List[float]]) -> Tuple[float, float]:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    @staticmethod
    def calculate_required_angle(current_angle: float, target_point: Tuple[float, float]) -> float:
        if abs(target_point[0]) < 1e-9 and abs(target_point[1]) < 1e-9:
            return 0.0

        target_angle = math.degrees(math.atan2(target_point[1], target_point[0]))

        current_normalized = current_angle % 360
        target_normalized = target_angle % 360

        # Calculate the minimal rotation angle
        diff = abs(target_normalized - current_normalized)
        rotation_angle = min(diff, 360 - diff)

        return rotation_angle

    def calculate_rotation_time(self, current_angle: float, target_point: Tuple[float, float],
                               camera_velocity: float) -> float:
        rotation_angle = self.calculate_required_angle(current_angle, target_point)
        return rotation_angle / camera_velocity if camera_velocity > 0 else 0

    def calculate_total_time_for_path(self, path: List[Tuple[float, float]],
                                     initial_angle: float,
                                     stage_velocity: float,
                                     camera_velocity: float) -> Tuple[float, List[float]]:
        current_angle = initial_angle
        total_time = 0.0
        times = [0.0]

        for i in range(1, len(path)):
            rotation_time = self.calculate_rotation_time(current_angle, path[i], camera_velocity)

            movement_distance = self.euclidean_distance(path[i-1], path[i])
            movement_time = movement_distance / stage_velocity if stage_velocity > 0 else 0

            current_angle = math.degrees(math.atan2(path[i][1], path[i][0]))

            segment_time = max(rotation_time, movement_time)
            total_time += segment_time
            times.append(total_time)
        
            if i < 5 or i > len(path) - 5:
                print(f"  Step {i}: From {path[i-1]} to {path[i]}")
                print(f"    Rotation: {rotation_time:.3f}s, Movement: {movement_time:.3f}s")
                print(f"    Segment time: {segment_time:.3f}s, Total: {total_time:.3f}s")

        return total_time, times

    def nearest_insertion_with_rotation(self, die_centers: List[Tuple[float, float]],
                                       start_pos: Tuple[float, float],
                                       initial_angle: float,
                                       stage_velocity: float,
                                       camera_velocity: float) -> List[Tuple[float, float]]:
     
        unvisited = list(range(len(die_centers)))
        tour = [0]  # Start with first die (index 0)
        unvisited.remove(0)

        # Start with the closest die to start position
        start_distances = [self.euclidean_distance(start_pos, die_centers[i]) for i in unvisited]
        if start_distances:
            closest_idx = unvisited[np.argmin(start_distances)]
            tour.append(closest_idx)
            unvisited.remove(closest_idx)

        while unvisited:
            best_cost_increase = float('inf')
            best_die_idx = -1
            best_position = -1

            # For each unvisited die
            for die_idx in unvisited:
                die_pos = die_centers[die_idx]

                # Find best position to insert this die
                for pos in range(len(tour)):
                    # Insert between tour[pos] and tour[(pos+1)%len(tour)]
                    if pos == len(tour) - 1:
                        # End of tour, would need to go back to start
                        continue

                    # Calculate insertion cost
                    current_from = die_centers[tour[pos]]
                    current_to = die_centers[tour[pos + 1]]

                    # New edges: from->new_die and new_die->to
                    original_distance = self.euclidean_distance(current_from, current_to)
                    new_distance = (self.euclidean_distance(current_from, die_pos) +
                                  self.euclidean_distance(die_pos, current_to))

                    # Rotation costs
                    if pos == 0:
                        # Need to consider initial angle for first segment
                        current_angle = initial_angle
                    else:
                        # Camera points to previous die
                        prev_die_pos = die_centers[tour[pos]]
                        current_angle = math.degrees(math.atan2(prev_die_pos[1], prev_die_pos[0]))

                    # Original rotation cost
                    original_rotation = self.calculate_rotation_time(current_angle, current_to, camera_velocity)

                    # New rotation costs
                    rotation_to_die = self.calculate_rotation_time(current_angle, die_pos, camera_velocity)
                    # After visiting die, camera points to die, then rotate to next
                    new_angle_after_die = math.degrees(math.atan2(die_pos[1], die_pos[0]))
                    rotation_die_to_next = self.calculate_rotation_time(new_angle_after_die, current_to, camera_velocity)

                    # Time costs
                    original_time = max(original_rotation, original_distance/stage_velocity)
                    new_time = (max(rotation_to_die, self.euclidean_distance(current_from, die_pos)/stage_velocity) +
                              max(rotation_die_to_next, self.euclidean_distance(die_pos, current_to)/stage_velocity))

                    cost_increase = new_time - original_time

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_die_idx = die_idx
                        best_position = pos + 1

            # Insert the best die at the best position
            if best_die_idx != -1:
                tour.insert(best_position, best_die_idx)
                unvisited.remove(best_die_idx)
            else:
                # Fallback: insert at end
                tour.append(unvisited.pop())

        # Convert indices to coordinates
        path = [start_pos] + [die_centers[i] for i in tour]
        return path

    def two_opt_with_rotation(self, path: List[Tuple[float, float]],
                             initial_angle: float,
                             stage_velocity: float,
                             camera_velocity: float) -> List[Tuple[float, float]]:
       
        best = path[:]
        best_time, _ = self.calculate_total_time_for_path(best, initial_angle, stage_velocity, camera_velocity)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1:
                        continue

                    # Create new path by reversing segment i to j
                    new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]

                    # Calculate total time for new path
                    new_time, _ = self.calculate_total_time_for_path(new_path, initial_angle, stage_velocity, camera_velocity)

                    if new_time < best_time - 1e-6:  # Small tolerance
                        best = new_path
                        best_time = new_time
                        improved = True
                        break
                if improved:
                    break

        return best

    def optimize_path(self, input_data: Dict[str, Any], use_2opt: bool = True) -> Dict[str, Any]:
      
        # Extract data
        start_pos = tuple(input_data["InitialPosition"])
        initial_angle = input_data["InitialAngle"]
        stage_velocity = input_data["StageVelocity"]
        camera_velocity = input_data["CameraVelocity"]  # deg/ms, convert to deg/s
        camera_velocity_deg_s = camera_velocity * 1000  # Convert from deg/ms to deg/s

        # Calculate die centers
        die_centers = []
        for die in input_data["Dies"]:
            center = self.calculate_die_center(die["Corners"])
            die_centers.append(center)

        print(f"Processing {len(die_centers)} dies...")
        print(f"Initial position: {start_pos}, Initial angle: {initial_angle}°")
        print(f"Stage velocity: {stage_velocity} mm/s")
        print(f"Camera velocity: {camera_velocity} deg/ms = {camera_velocity_deg_s} deg/s")

        # Step 1: Nearest Insertion algorithm considering rotation
        print("\nRunning Nearest Insertion algorithm...")
        path = self.nearest_insertion_with_rotation(
            die_centers, start_pos, initial_angle, stage_velocity, camera_velocity_deg_s
        )

        # Calculate initial total time
        initial_time, _ = self.calculate_total_time_for_path(
            path, initial_angle, stage_velocity, camera_velocity_deg_s
        )
        print(f"Initial path time: {initial_time:.3f} seconds")

        # Step 2: 2-opt optimization
        if use_2opt and len(path) > 3:
            print("Applying 2-opt optimization...")
            path = self.two_opt_with_rotation(
                path, initial_angle, stage_velocity, camera_velocity_deg_s
            )

        # Final calculation
        total_time, segment_times = self.calculate_total_time_for_path(
            path, initial_angle, stage_velocity, camera_velocity_deg_s
        )

        # Format output
        formatted_path = [list(p) for p in path]

        # Prepare detailed output
        output_data = {
            "TotalTime": total_time,  # in seconds
            "Path": formatted_path,
            "SegmentTimes": segment_times  # Cumulative times at each point
        }

        print(f"\n✅ Optimization complete!")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Path length: {len(path)} points")

        return output_data


def process_milestone2():
   
   
    # SPECIFIC FILE PATH - You can change this to your desired path
    input_file_path = r"C:\Users\kla_user\Downloads\planner\Input_Milestone2_Testcase2.json"
   
    # Define output file path (in the same directory with Output_ prefix)
    output_dir = os.path.dirname(input_file_path)
    input_filename = os.path.basename(input_file_path)
    output_filename = input_filename.replace('Input_Milestone2_Testcase', 'TestCase_2_')
    output_file_path = os.path.join(output_dir, output_filename)
   
    print("=" * 80)
    print("Wafer Die Path Optimization with Camera Rotation")
    print("=" * 80)
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print("=" * 80)
   
    try:
        # Read input from JSON file
        with open(input_file_path, 'r') as f:
            input_data = json.load(f)
       
        # Create optimizer
        optimizer = DiePathOptimizer()
       
        # Optimize path (using 2-opt by default)
        output_data = optimizer.optimize_path(input_data, use_2opt=True)
       
        # Save output to JSON file
        with open(output_file_path, 'w') as f:
            # Save only required fields
            simplified_output = {
                "TotalTime": output_data["TotalTime"],
                "Path": output_data["Path"]
            }
            json.dump(simplified_output, f, indent=2)
       
        print(f"\n✅ Results saved to: {output_file_path}")
       
        # Show final summary
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("-" * 40)
        print(f"Total Time: {output_data['TotalTime']:.6f} seconds")
        print(f"Number of points in path: {len(output_data['Path'])}")
        print("=" * 80)
       
        return output_data
       
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file_path}")
        print("Please check the file path and try again.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in input file: {e}")
        return None
    except KeyError as e:
        print(f"❌ Error: Missing key in input data: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the optimization with the specific file path
    process_milestone2()

