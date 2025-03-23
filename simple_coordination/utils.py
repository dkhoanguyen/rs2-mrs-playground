import numpy as np
from typing import Any
import yaml

# Read YAML file and extract data
def load_yaml_data(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    
    task_id = [task["id"] for task in data["tasks"]]
    task_location = np.array([task["location"] for task in data["tasks"]])
    task_times = np.array([task["time"] for task in data["tasks"]])
    
    return task_id, task_location, task_times

def compute_travel_time(pos1, pos2, velocity=1.0):
    """Computes travel time based on Euclidean distance and velocity."""
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return distance / velocity


def construct_task_matrix_dict(
        robot_id: list,
        task_id: list,
        tasks: np.ndarray,
        initial_pos: np.ndarray,
        task_times: dict,
        robot_velocity: float = 1.0,
        output_type: str = "matrix"):
    H_dict = {}
    H_matrix = np.empty((len(robot_id),len(task_id)))

    for i, robot in enumerate(robot_id):
        H_dict[robot] = {}
        for j, task in enumerate(tasks):
            travel_time = compute_travel_time(
                initial_pos[i, :2], task[:2], robot_velocity[i])
            # Default to 0 if task time is not found
            task_time = task_times[j]
            H_dict[robot][task_id[j]] = travel_time + task_time
            H_matrix[i,j] = travel_time + task_time
    if output_type == "matrix":
        return H_matrix
    return H_dict


def print_task_matrix(H: dict):
    """Utility function to print the task matrix in a readable format."""
    tasks = list(next(iter(H.values())).keys())
    print("\t" + "\t".join(tasks))
    for robot, task_times in H.items():
        print(f"{robot}\t" +
              "\t".join(f"{task_times[task]:.2f}" for task in tasks))
