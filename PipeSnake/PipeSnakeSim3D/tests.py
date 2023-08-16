import matplotlib.pyplot as plt
import numpy as np
from pipe import Pipe
from robot import Robot

""" Write tests here """

def test1():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    p = Pipe(0.2, [0,0,0], [1, 0, 0])
    p.add_straight_pipe()
    p.add_turn([1., 1., 0.])
    p.add_straight_pipe(1.6)
    p.add_turn([0., 1., 0.])
    p.add_straight_pipe()
    p.add_turn([0.2, 0., 0.])
    p.add_straight_pipe()
    p.add_turn([0., 1., 1.])
    p.add_straight_pipe()
    p.add_turn([-1., 0., 0.])
    p.add_straight_pipe()
    p.add_turn([0.3, 1., 2.0])
    p.add_straight_pipe()
    plt.plot(p.pts[0], p.pts[1], p.pts[2])
    
    plt.title("PipeSnake Visualization")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_zlabel("z (meters)")
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,2,1,1]))
    ax.set_aspect('equal')
    plt.show()
    plt.close(fig=fig)

def test2():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    p = Pipe(0.2, [0,0,0], [1, 0, 1])
    p.add_straight_pipe()
    p.add_turn_euler([0, 0, np.pi / 2])
    p.add_straight_pipe(1.0)
    p.add_turn([0.,0.,1.])
    p.add_straight_pipe(0.5)
    p.add_turn([1.,1.,1.])
    p.add_straight_pipe(0.8)
    plt.plot(p.pts[0], p.pts[1], p.pts[2])
    
    # Create Bot model
    b = Robot((0, 0, 0), p)
    b.init_bot()
    # Plot bot
    n_wheels = len(b.wheels)
    for i in range(n_wheels):
        ax.plot_surface(b.wheels[i][0], b.wheels[i][1], b.wheels[i][2], color="gray", alpha = 0.9, linewidth=5)
    n_lines = len(b.lines)
    for i in range(n_lines):
        ax.plot3D(b.lines[i][0], b.lines[i][1], b.lines[i][2], linewidth=3, color="red")

    plt.title("PipeSnake Visualization")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_zlabel("z (meters)")
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,2,1,1]))
    ax.set_aspect('equal')
    plt.show()
    plt.close(fig=fig)

def main():
    test1()
    # test2()
    
if __name__ == "__main__":
    main()