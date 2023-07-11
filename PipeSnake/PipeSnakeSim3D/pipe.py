import matplotlib.pyplot as plt
import numpy as np
import transforms3d as t3d
import math

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
        self.resolution = 5
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
        
        print("Straight pipe added successfully, Direction: " + self.direction)
      
    # adds turn to the pipe in the specified direction
    # direction is a string representing what direction to turn to, can be:
    #   +x, -x, +y, -y, +z, -z       
    def add_turn(self, turn_to):
        # get cross section coords at end of pipe, then rotate it at increments of pi/60, plotting each one, then plotting the surface graph
        # turns can be pi/2 radians or pi radians
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
                def get_transformation1(pose):
                    T_ru = np.eye(4)
                    T_ru[0:3, 0:3] = t3d.quaternions.quat2mat([pose[6],pose[3],pose[4],pose[5]])
                    T_ru[0:3, 3] = [pose[0], pose[1], pose[2]]
    
                    return T_ru
                
                def transform_cross(rotate, translate):
                    pose = [x, y, 0, 0, 0, math.sin(rotate/2), math.cos(rotate/2)]
                    T = get_transformation1(pose)
                    x_cross_copy = x_cross.copy()
                    y_cross_copy = y_cross.copy()
                    z_cross_copy = z_cross.copy()
                    for i in range(self.resolution):
                        for j in range(self.resolution):
                            p = np.matmul(T, [x_cross[i][j], y_cross[i][j], z_cross[i][j], 1])
                            x_cross_copy[i][j] = p[0]
                            y_cross_copy[i][j] = p[1]
                            z_cross_copy[i][j] = p[2]
                
                    self.x = np.append(self.x, x_cross_copy, axis=1)
                    self.y = np.append(self.y, y_cross_copy, axis=1)
                    self.z = np.append(self.z, z_cross_copy, axis=1)
                
                theta = -np.pi/2
                transform_cross(theta/2, 0)
                transform_cross(theta, 0.33)
                self.start_pt = (x + 0.33, y + 0.33, z)
                self.direction = "+x"
                
                # Transforms = []
                # for i in range(self.resolution):
                #     pose = [x + translate[i], y + translate[i], 0, 0, 0, math.sin(rotate[i]/2), math.cos(rotate[i]/2)]
                #     Transforms.append(get_transformation1(pose))

                # for T in Transforms:
                #     for i in range(self.resolution):
                #         for j in range(self.resolution):
                #             p = np.array([x_cross[i][j], y_cross[i][j], z_cross[i][j], 1])
                #             p = np.matmul(T, p)
                #             x_cross[i][j] = p[0]
                #             y_cross[i][j] = p[1]
                #             z_cross[i][j] = p[2]             
        elif turn_to == "-x":
            print("Unimplemented")
        elif turn_to == "+z":
            print("Unimplemented")
        elif turn_to == "-z":
            print("Unimplemented")