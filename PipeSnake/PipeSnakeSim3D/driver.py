import matplotlib.pyplot as plt
import numpy as np
from pipe import Pipe
from robot import Robot

""" The current front-end of the visualization.

Call methods from Pipe and Robot to build the pipe environment and the pipesnake model.

TODO:
    - GUI
    - Anything to improve usability
"""

def main():
    # Initializing matplotlib variables
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    # Create Pipe environment and plot
    p = Pipe(0.2, [0,0,0], [0, 1, 0])
    p.add_straight_pipe(1.0)
    p.add_turn_euler([0, 0, np.pi / 2])
    plt.plot(p.pts[0], p.pts[1], p.pts[2])
    
    # Create Bot model and plot
    b = Robot((0, 0, 0), p)
    b.init_bot()
    n_wheels = len(b.wheels)
    for i in range(n_wheels):
        ax.plot_surface(b.wheels[i][0], b.wheels[i][1], b.wheels[i][2], color="gray", alpha = 0.9, linewidth=5)
    n_lines = len(b.lines)
    for i in range(n_lines):
        ax.plot3D(b.lines[i][0], b.lines[i][1], b.lines[i][2], linewidth=3, color="red")
    
    # Sets up plot with labels and aspect
    plt.title("PipeSnake Visualization")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_zlabel("z (meters)")
    ax.set_aspect('equal')
    plt.show()
    plt.close(fig=fig)

if __name__ == "__main__":
    main()
    