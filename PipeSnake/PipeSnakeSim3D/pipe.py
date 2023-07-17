import matplotlib.pyplot as plt
import numpy as np
import transforms3d as t3d
import math
import pandas as pd

class Pipe:
    def __init__(self, radius, start_pt):
        self.radius = radius
        self.x = np.zeros((1,1))
        self.y = np.zeros((1,1))
        self.z = np.zeros((1,1))
        self.start_pt = start_pt
        self.direction = "+y"       # direction specifies what axis to extend the pipe and in what direction along the axis
                                    # default is +y
        self.resolution = 20        # resolutions specifies how "smooth" the pipe will look, higher values lead to more smoothness but more processing
        self.length = 1.53          # length is the length of all straight pipes
        self.steps = 6              # steps is the number of rings in turns, can affect the visibility of the robot in turns
            
    # Draws a straight pipe of length 1.53 meters of specified radius
    # units of the pipe is meters
    def add_straight_pipe(self):
        # get point to start the pipe
        center_x = self.start_pt[0]
        center_y = self.start_pt[1]
        center_z = self.start_pt[2]
        
        # form x, y, z point arrays
        x_grid = np.array(0)
        y_grid = np.array(0)
        z_grid = np.array(0)
        prev_start = self.start_pt
        if self.direction == "+y":
            y_pts = np.linspace(center_y, center_y + self.length, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y + self.length, center_z)
            
        elif self.direction == "-y":
            y_pts = np.linspace(center_y - self.length, center_y, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y - self.length, center_z)
            
        elif self.direction == "+x":
            x_pts = np.linspace(center_x, center_x + self.length, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x + self.length, center_y, center_z)
             
        elif self.direction == "-x":
            x_pts = np.linspace(center_x - self.length, center_x, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x - self.length, center_y, center_z)
            
        elif self.direction == "+z":
            z_pts = np.linspace(center_z, center_z + self.length, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z + self.length)
            
        elif self.direction == "-z":
            z_pts = np.linspace(center_z - self.length, center_z, self.resolution)
            theta = np.linspace(0, 2*np.pi,self.resolution)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z - self.length)
        
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
            
        # print(self.x.shape)
        # print(self.y.shape)
        # print(self.z.shape)
        
        print("Straight pipe added successfully, Direction: " + self.direction + ", Started from: " + str(prev_start), "Ended at: " + str(self.start_pt))
    
    
    # adds 90 degree turn to the pipe in the specified direction
    # direction is a string representing what direction to turn to, can be:
    #   +x, -x, +y, -y, +z, -z       
    def add_turn(self, turn_to):
        if (self.direction == "+y" or self.direction == "-y") and (turn_to == "+y" or turn_to == "-y"):
            print("Invalid Turn: Can't turn from a y direction to a y direction")
            return
        if (self.direction == "+x" or self.direction == "-x") and (turn_to == "+x" or turn_to == "-x"):
            print("Invalid Turn: Can't turn from a x direction to a x direction")
            return
        if (self.direction == "+z" or self.direction == "-z") and (turn_to == "+z" or turn_to == "-z"):
            print("Invalid Turn: Can't turn from a z direction to a z direction")
            return
        x, y, z = self.start_pt
         
        # transforms x, y, and z points using transformation matrix T
        def transform(T, x_pts, y_pts, z_pts):
            for i in range(self.resolution):
                for j in range(self.resolution):
                    p = np.matmul(T, [x_cross[i][j], y_cross[i][j], z_cross[i][j], 1])
                    x_pts[i][j] = p[0]
                    y_pts[i][j] = p[1]
                    z_pts[i][j] = p[2]
            return x_pts, y_pts, z_pts

        # transforms the cross section
        # rotates by angle (in radians) provided in rotate parameter
        # translate x, y, and z by translate parameter
        def transform_cross(rotate):
            # ROTATE
            rot_x = 0
            rot_y = 0
            rot_z = 0
            
            if self.direction == "+y":
                if turn_to == "+x" or turn_to == "-x":
                    rot_z = rotate
                elif turn_to == "+z" or turn_to == "-z":
                    rot_x = rotate
            
            R_z = np.array([
                [np.cos(rot_z), -1 * np.sin(rot_z), 0],
                [np.sin(rot_z), np.cos(rot_z), 0],
                [0, 0, 1]
            ])
            R_y = np.array([
                [np.cos(rot_y), 0, np.sin(rot_y)],
                [0, 1, 0],
                [-1 * np.sin(rot_y), 0, np.cos(rot_y)]
            ])
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(rot_x), -1 * np.sin(rot_x)],
                [0, np.sin(rot_x), np.cos(rot_x)]
            ])
            R = np.eye(4)
            M = R_z @ R_y @ R_x
            R[0:3, 0:3] = M
            
            x_cross_copy = x_cross.copy()
            y_cross_copy = y_cross.copy()
            z_cross_copy = z_cross.copy()
            x_cross_copy, y_cross_copy, z_cross_copy = transform(R, x_cross_copy, y_cross_copy, z_cross_copy)
            
            # TRANSLATE
            # When points rotate, there's also some translation, so position of rotated points must be corrected
            # Create a translation matrix using arbitrary point on rotated cross-section (p2) and point on
            # cross-section before rotation (p1)
            p1 = []
            p2 = []
            if self.direction == "+y":
                if turn_to == "+x":
                    p1 = [x_cross[0][0], y_cross[0][0], z_cross[0][0]]
                    p2 = [x_cross_copy[0][0], y_cross_copy[0][0], z_cross_copy[0][0]]
                if turn_to == "-x":
                    p1 = [x_cross[0][self.resolution // 2], y_cross[0][self.resolution // 2], z_cross[0][0]]
                    p2 = [x_cross_copy[0][self.resolution // 2], y_cross_copy[0][self.resolution // 2], z_cross_copy[0][0]]
                if turn_to == "+z":
                    p1 = [x_cross[0][self.resolution // 4], y_cross[0][self.resolution // 4], z_cross[0][self.resolution // 4]]
                    p2 = [x_cross_copy[0][self.resolution // 4], y_cross_copy[0][self.resolution // 4], z_cross_copy[0][self.resolution // 4]]
                if turn_to == "-z":
                    p1 = [x_cross[0][(3 * self.resolution) // 4], y_cross[0][(3 * self.resolution) // 4], z_cross[0][(3 * self.resolution) // 4]]
                    p2 = [x_cross_copy[0][(3 * self.resolution) // 4], y_cross_copy[0][(3 * self.resolution) // 4], z_cross_copy[0][(3 * self.resolution) // 4]]
            TRNFM = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]
            # translate points
            x_cross_copy += TRNFM[0]
            y_cross_copy += TRNFM[1]
            z_cross_copy += TRNFM[2]
            # # update local x, y, z point arrays
            self.x = np.append(self.x, x_cross_copy, axis=1)
            self.y = np.append(self.y, y_cross_copy, axis=1)
            self.z = np.append(self.z, z_cross_copy, axis=1)
        
        if self.direction == "+y":
            # collect points of cross section at end of pipe 
            y_pts = np.repeat(y, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, y_cross=np.meshgrid(theta, y_pts)
            x_cross = self.radius*np.cos(theta_grid) + x
            z_cross = self.radius*np.sin(theta_grid) + z
            
            # NOTE: COUNTER-CLOCKWISE rotations are POSITIVE, CLOCKWISE rotations are NEGATIVE
            if turn_to == "+x":
                # rotate points of cross section in small increments
                theta = -np.pi/2 
                rotates = np.linspace(theta, 0, self.steps)[::-1]
                for i in range(self.steps):
                    transform_cross(rotates[i])
                self.start_pt = (x + self.radius, y + self.radius, z)
                self.direction = "+x"
                print("Successful turn from +y to +x, Started from: " + str((x, y, z)) + ", Ended at: " + str(self.start_pt))
            elif turn_to == "-x":
                theta = np.pi/2
                rotates = np.linspace(0, theta, self.steps)
                for i in range(self.steps):
                    transform_cross(rotates[i])
                self.start_pt = (x - self.radius, y + self.radius, z)
                self.direction = "-x"
                print("Successful turn from +y to -x, Started from: " + str((x, y, z)) + ", Ended at: " + str(self.start_pt))
            elif turn_to == "+z":
                theta = np.pi/2
                rotates = np.linspace(theta, 0, self.steps)[::-1]
                for i in range(self.steps):
                    transform_cross(rotates[i])
                self.start_pt = (x, y + self.radius, z + self.radius)
                self.direction = "+z"
                print("Successful turn from +y to +z, Started from: " + str((x, y, z)) + ", Ended at: " + str(self.start_pt))
            elif turn_to == "-z":
                theta = -np.pi/2
                rotates = np.linspace(0, theta, self.steps)
                for i in range(self.steps):
                    transform_cross(rotates[i])
                self.start_pt = (x, y + self.radius, z - self.radius)
                self.direction = "-z"
                print("Successful turn from +y to -z, Started from: " + str((x, y, z)) + ", Ended at: " + str(self.start_pt))
                
                    
        