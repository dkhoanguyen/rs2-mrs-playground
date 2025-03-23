import irsim  # initialize the environment with the configuration file
import irsim.world
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def main():
    env = irsim.make('coordination_world.yaml')
    plt.gcf().set_size_inches(7, 7)
    env.render()
    env.end(ending_time=10)

    # Simple setup
    env = irsim.make('coordination_world.yaml')
    plt.gcf().set_size_inches(7, 7)

    # Define tasks (goal location)
    # Suppose we have 2 robots and 2 tasks to do
    task_id, task_location, task_times = load_yaml_data("task.yaml")

    # What if robots have different speed
    robot_velocity = np.array([
        0.5, # Robot 0 - move slower
        1.5  # Robot 1 - move faster
    ])

    # Populate relevant data first
    robot_id = []
    initial_pos = np.empty((0, 3))
    robot: irsim.world.ObjectBase
    for id, robot in enumerate(env.robot_list):
        robot_id.append(robot.get_info().id)
        initial_pos = np.vstack((initial_pos, robot.state.T))
        robot.vel_max = np.array([robot_velocity[id], 1]).reshape((2, 1))

    # Assign tasks to robots based on the order in the task matrix
    # regardless of execution time and travel time
    for i, robot in enumerate(env.robot_list):
        robot.set_goal(task_location[i, :].reshape((3,1)))

    for i in range(300):  # run the simulation for 300 steps
        robot: irsim.world.ObjectBase
        for robot in env.robot_list:
            robot.step()
        env.render()  # render the environment
        if env.done():
            break  # check if the simulation is done
        # Plot the iteration number in the lower-right corner
        plt.text(0.975, 0.025, f"Step: {i}", fontsize=12, color='black',
                transform=plt.gca().transAxes, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    print(f"Total steps: {i}")
    env.end()  # close the environment


if __name__ == "__main__":
    main()
