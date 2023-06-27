import matplotlib.pyplot as plt
import math
import statistics
import numpy as np

# Given 8 points/vectors(?) and a time, draw the pipe
# The 8 points outline part of the pipe's cross section, while the time represents the length of the pipe
# Points are given as a tuple of x and y
def graphPipe(points, time):
    def data_for_cylinder_along_y(center_x,center_z,radius,length_y):
        y = np.linspace(0, length_y, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, y_grid=np.meshgrid(theta, y)
        x_grid = radius*np.cos(theta_grid) + center_x
        z_grid = radius*np.sin(theta_grid) + center_z
        return x_grid,y_grid,z_grid
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.title("Pipe Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    x1 = []
    y1 = [0] * len(points)
    z1 = []
    for a,b in points:
        x1.append(a)
        z1.append(b)
    ax.scatter3D(x1,y1,z1, color="red")
    
    rads = []
    for a,b in points:
        rads.append(math.sqrt(a**2 + b**2))
    rads.sort()
    radius = statistics.median(rads)
    x2, y2, z2 = data_for_cylinder_along_y(0,0,radius,time)
    ax.plot_surface(x2, y2, z2, alpha=0.5)
    
    plt.show()
    plt.close(fig=fig)
    
graphPipe([(0.,3.), (-2.,2.236), (-3,0)], 10)   