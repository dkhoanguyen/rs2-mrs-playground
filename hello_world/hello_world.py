import irsim  # initialize the environment with the configuration file
import matplotlib.pyplot as plt

import irsim.world.robots

def main():
    # Simple setup
    env = irsim.make('hello_world.yaml')
    plt.gcf().set_size_inches(5, 5) 
    
    #
    for i in range(300):  # run the simulation for 300 steps
        robot: irsim.world.ObjectBase
        for robot in env.robot_list:
            robot.step()
        env.render()  # render the environment
        if env.done():
            break  # check if the simulation is done
    env.end()  # close the environment

if __name__ == "__main__":
    main()
