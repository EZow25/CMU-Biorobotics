import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pipe import Pipe

def main():
    p = Pipe(5, (0,0,0))      
    p.add_straight_pipe(50)
    p.add_straight_pipe(70)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 1, 0.5, 1]))
    ax.plot_surface(p.x, p.y, p.z, alpha=0.5)
    plt.title("Ultrasound Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    plt.close(fig=fig)

if __name__ == "__main__":
    main()
    