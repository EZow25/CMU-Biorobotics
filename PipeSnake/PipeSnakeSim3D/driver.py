import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pipe import Pipe

def main():
    # a = np.array([1,2,3,4,5])
    # b = np.array([6,7,8,9,10])
    # c,d = np.meshgrid(a, b)
    # print(c)
    # print(d)
    
    p = Pipe(0.1625, (0,0,0))      
    p.add_straight_pipe()
    p.add_turn("-z")
    p.add_straight_pipe()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1,1,1,1]))
    ax.plot_wireframe(p.x, p.y, p.z, alpha=0.5)
    plt.title("PipeSnake Visualization")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_zlabel("z (meters)")
    ax.set_aspect('equal')
    plt.show()
    plt.close(fig=fig)

if __name__ == "__main__":
    main()
    