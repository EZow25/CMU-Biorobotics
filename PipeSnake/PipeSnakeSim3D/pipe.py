import matplotlib.pyplot as plt
import numpy as np

class Pipe:
    def __init__(self, radius):
        self.radius = radius
        self.x = np.zeros((1,1))
        self.y = np.zeros((1,1))
        self.z = np.zeros((1,1))
    
    # Draws a straight pipe
    # center is the (x,y,z) of the start of the center of the pipe
    # radius is the radius of the pipe
    # length is the length of the pipe
    def add_straight_pipe(self, center, length):
        center_x = center[0]
        center_y = center[1]
        center_z = center[2]
        y_pts = np.linspace(center_y, center_y + length,30)
        theta = np.linspace(0, 2*np.pi,30)
        theta_grid, y_grid=np.meshgrid(theta, y_pts)
        x_grid = self.radius*np.cos(theta_grid) + center_x
        z_grid = self.radius*np.sin(theta_grid) + center_z
        if np.array_equal(self.x, np.zeros((1,1))):
            self.x = x_grid
        else:
            self.x = np.append(self.x, x_grid, axis=1)
            
        if np.array_equal(self.y, np.zeros((1,1))):
            self.y = y_grid
        else:
            self.y = np.append(self.y, y_grid, axis=1)
            
        if np.array_equal(self.z, np.zeros((1,1))):
            self.z = z_grid
        else:
            self.z = np.append(self.z, z_grid, axis=1)