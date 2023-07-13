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
        self.direction = "+y"
        # direction specifies what axis to extend the pipe and in what direction along the axis
        # default is +y
        self.resolution = 30
        # resolutions specifies how "smooth" the pipe will look, higher values lead to more smoothness but more processing
            
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
        if self.direction == "+y":
            y_pts = np.linspace(center_y, center_y + 1.53, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y + 1.53, center_z)
            
        elif self.direction == "-y":
            y_pts = np.linspace(center_y - 1.53, center_y, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, y_grid=np.meshgrid(theta, y_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x, center_y - 1.53, center_z)
            
        elif self.direction == "+x":
            x_pts = np.linspace(center_x, center_x + 1.53, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x + 1.53, center_y, center_z)
             
        elif self.direction == "-x":
            x_pts = np.linspace(center_x - 1.53, center_x, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, x_grid=np.meshgrid(theta, x_pts)
            y_grid = self.radius*np.cos(theta_grid) + center_y
            z_grid = self.radius*np.sin(theta_grid) + center_z
            
            self.start_pt = (center_x - 1.53, center_y, center_z)
            
        elif self.direction == "+z":
            z_pts = np.linspace(center_z, center_z + 1.53, self.resolution)
            theta = np.linspace(0, 2*np.pi, self.resolution)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z + 1.53)
            
        elif self.direction == "-z":
            z_pts = np.linspace(center_z - 1.53, center_z, self.resolution)
            theta = np.linspace(0, 2*np.pi,self.resolution)
            theta_grid, z_grid=np.meshgrid(theta, z_pts)
            x_grid = self.radius*np.cos(theta_grid) + center_x
            y_grid = self.radius*np.sin(theta_grid) + center_y
            
            self.start_pt = (center_x, center_y, center_z - 1.53)
        
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
        
        print("Straight pipe added successfully, Direction: " + self.direction + ", Start_pt: " + str(self.start_pt))
      
    # adds turn to the pipe in the specified direction
    # direction is a string representing what direction to turn to, can be:
    #   +x, -x, +y, -y, +z, -z       
    def add_turn(self, turn_to):
        x, y, z = self.start_pt
        
        if turn_to == "+y":
            print("Unimplemented")
        elif turn_to == "-y":
            print("Unimplemented")
        elif turn_to == "+x":
            if self.direction == "+y":
                # collect points of cross section at end of pipe 
                y_pts = np.repeat(y, self.resolution)
                theta = np.linspace(0, 2*np.pi, self.resolution)
                theta_grid, y_cross=np.meshgrid(theta, y_pts)
                x_cross = self.radius*np.cos(theta_grid) + x
                z_cross = self.radius*np.sin(theta_grid) + z
                   
                # rotate points of cross section in small increments
                
                # builds transformation matrix
                # pose = [x_translation, y_translation, z_translation, quaternion]
                def get_transformation1(pose):
                    T_ru = np.eye(4)
                    T_ru[0:3, 0:3] = t3d.quaternions.quat2mat([pose[6],pose[3],pose[4],pose[5]])
                    T_ru[0:3, 3] = [pose[0], pose[1], pose[2]]
    
                    return T_ru
                
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
                def transform_cross(rotate, translate):
                    # ROTATE
                    R = [0, 0, 0, 0, 0, math.sin(rotate/2), math.cos(rotate/2)]
                    T0 = get_transformation1(R)
                    x_cross_copy = x_cross.copy()
                    y_cross_copy = y_cross.copy()
                    z_cross_copy = z_cross.copy()
                    x_cross_copy, y_cross_copy, z_cross_copy = transform(T0, x_cross_copy, y_cross_copy, z_cross_copy)
                    # TRANSLATE
                    # reference point
                    p1 = [x_cross[0][0], y_cross[0][0], z_cross[0][0]]
                    # find index of point of smallest z value
                    # p2_idx = pd.Series(z_cross_copy[0]).idxmin()
                    p2 = [x_cross_copy[0][0], y_cross_copy[0][0], z_cross_copy[0][0]]
                    # p2 = [x_cross[0][p1_idx], y_cross[0][p1_idx], z_cross[0][p1_idx]]
                    
                    # find translation from p2 to p1, then translate based on parameter provided
                    TRNFM = [p1[0] - p2[0] + translate, p1[1] - p2[1] + translate, p1[2] - p2[2]]
                    x_cross_copy += TRNFM[0]
                    y_cross_copy += TRNFM[1]
                    z_cross_copy += TRNFM[2]
                    self.x = np.append(self.x, x_cross_copy, axis=1)
                    self.y = np.append(self.y, y_cross_copy, axis=1)
                    self.z = np.append(self.z, z_cross_copy, axis=1)
                
                theta = -np.pi/2
                # transform_cross(theta/2, 0)
                # steps stores how many rings will be in the curve
                steps = 10
                translates = np.linspace(0, 0.33, steps) 
                rotates = np.linspace(theta, 0, steps)[::-1]
                for i in range(steps):
                    transform_cross(rotates[i], translates[i])
                self.start_pt = (x + 0.33, y + 0.33, z)
                self.direction = "+x"
                print("Successful turn from +y to +x")
                
                # NOTES
                # The inner edge of the turn is curved incorrectly
                # Instead of even spacing for translates, try exponentially changing the step spacing
                # Exponential changes for rotates?
                # Both?
                # The middles of the cross sections need to follow some curve
                # Rotation and translate step spaces appear to both increase exponentially, then decrease, then increase again
                # but the max step spaces don't appear to be too drastic
                # STL files?
                             
        elif turn_to == "-x":
            print("Unimplemented")
        elif turn_to == "+z":
            print("Unimplemented")
        elif turn_to == "-z":
            print("Unimplemented")