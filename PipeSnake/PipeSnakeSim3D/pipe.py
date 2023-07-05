import matplotlib.pyplot as plt
import numpy as np
import transforms3d as t3d

class Pipe:
    def __init__(self, radius, start_pt):
        self.radius = radius
        self.x = np.zeros((1,1))
        self.y = np.zeros((1,1))
        self.z = np.zeros((1,1))
        self.start_pt = start_pt
        self.direction = "+y"
        # direction specifies what axis to extend the pipe and in what direction along the axis
        # default is +y
            
    # Draws a straight pipe
    # length is the length of the pipe
    def add_straight_pipe(self, length):
        # get point to start the pipe
        center_x = self.start_pt[0]
        center_y = self.start_pt[1]
        center_z = self.start_pt[2]
        
        # form x, y, z point arrays
        x_grid = np.array(0)
        y_grid = np.array(0)
        z_grid = np.array(0)
        if self.direction == "+y":
            y_pts = np.linspace(center_y, center_y + length, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y + length, center_z)
            
        elif self.direction == "-y":
            y_pts = np.linspace(center_y - length, center_y, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y - length, center_z)
            
        elif self.direction == "+x":
            x_pts = np.linspace(center_x, center_x + length, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x + length, center_y, center_z)
             
        elif self.direction == "-x":
            x_pts = np.linspace(center_x - length, center_x, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x - length, center_y, center_z)
            
        elif self.direction == "+z":
            z_pts = np.linspace(center_z, center_z + length, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z + length)
            
        elif self.direction == "-z":
            z_pts = np.linspace(center_z - length, center_z, 30)
            theta = np.linspace(0, 2*np.pi,30)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z - length)
        
        # update local vars for x, y, z points
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