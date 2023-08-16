import numpy as np
import transforms3d as t3d

""" Calculates all points for plotting the pipe environment. 
Each pipe is represented as circular cross sections placed in the shape of either a straight pipe or a curved pipe.
The pipe environment can be thought of as a network of pipe, with each section being either a straight pipe or a curved pipe.

TODO:
    - Optimize code to vectorize instead of using for loops.
    - Implement bendy pipes.
    - Implement T shaped pipes.
    - There's some extra lines in the viz connecting each section, find a way to remove
"""

class Pipe:
    """ This represents the pipe environment.
    
    Attributes:
        radius:
          The radius of cross-section circles. Type float.
        pts:
          The pts along all the cross-section circles. 3 x n numpy array, first row is x values, second is y values, last is z values.
        end_pt:
          The point at center of the cross-section circle at the very end of the pipe environment. Array of 3 floats.
        start_direction:
          Unit vector representing the direction of the first pipe section. Used for orienting the robot. Type numpy array of 3 floats.
        end_direction:
          Unit Vector representing the direction of the last pipe section. Type numpy array of 3 floats.
        resolution:
          Represents the number of points along all individual circles. It also represents the number of circles in straight pipes.
          Type integer.
        curved_res:
          Represents the number of circles in curved sections. This can affect the visibility of the robot as it traverses the curves.
          Type integer.
        turn_radius:
          The turning radius of curves. Type float.
    """
    
    def __init__(self, radius: float, start_pt: list[float], direction: list[float]):
        """ Initializes instance of Pipe with radius, start_pt and direction.

        Args:
            radius (float):
              Radius of cross-section circles.
            start_pt (list[float]):
              Point to start building the pipe from.
            direction (list[float]):
              The starting direction the pipe points in.
        """
        
        self.radius = radius
        self.pts = None
        self.end_pt = np.array(start_pt)
        self.start_direction = np.array(direction) / np.linalg.norm(direction)
        self.end_direction = np.array(direction) / np.linalg.norm(direction)
        self.resolution = 11
        self.curved_res = 6
        self.turn_radius = 0.5

    def build_circle(self, center: list[float], direction: list[float], radius: float) -> list[list[float]]:
        """ Calculates the points needed to plot a circle.

        Args:
            center (list[float]): 
              The center point of the circle.
            direction (list[float]):
              The normal unit vector of the plane the circle lies on.
            radius (float):
              The radius of the circle.

        Returns:
            list[list[float]]:
              The circle. Represented as a 3 x self.resolution array. The first row holds the x values, the second row holds the y values,
              the third row represents the z values.
        """
        
        # Finds 2 unit vectors, both orthogonal to the direction vector, which both lie on the plane the first cross-section circle lies on.
        # Essentially, these 2 unit vectors will define the plane.
        
        # Build p1, the first orthogonal vector.
        # For finding p1, do the approach described in 
        # https://math.stackexchange.com/questions/133177/finding-a-unit-vector-perpendicular-to-another-vector 
        center = np.array(center)
        direction = np.array(direction)
        m = 0
        while m < 2 and direction[m] == 0:
            m += 1
        n = 0
        if n == m:
            n += 1
        p1 = [0, 0, 0]
        p1[n] = direction[m]
        p1[m] = -1 * direction[n]
        p1 = np.array(p1) / np.linalg.norm(p1)
        
        # Build p2, which is orthogonal to p1 and the direction vector
        p2 = np.cross(direction, p1)
        p2 /= np.linalg.norm(p2)
        
        p1 = np.transpose(p1.reshape(1, 3))
        p1 = np.tile(p1, (1, self.resolution))
        p2 = np.transpose(p2.reshape(1, 3))
        p2 = np.tile(p2, (1, self.resolution))
        
        # Build first cross-section circle, on plane defined by p1 and p2
        cross = np.linspace(0, 2 * np.pi, self.resolution)
        cross = np.tile(cross, (3, 1))
        center = np.transpose(center.reshape(1, 3))
        center = np.tile(center, (1, self.resolution))
        
        # To calculate points, use the equation from
        # https://math.stackexchange.com/questions/3422062/circle-parameterization 
        cross = center + (radius * ((p1 * np.cos(cross)) + (p2 * np.sin(cross))))
        
        return cross
         
    def add_straight_pipe(self, length=1.53) -> None:
        """ Plots a straight pipe.
        
        We first plot the first cross-section circle in the straight pipe.
        From there, we translate subsequent cross-sections along the end_direction vector to get new cross-sections.
        This is continued until we have a line of cross-sections of length matching the length parameter.
        
        Args:
            length (float, optional): The length of the straight pipe. Defaults to 1.53.
        """
        
        prev_start = self.end_pt  # Saving the start point for the success message
        first_cross = self.build_circle(self.end_pt, self.end_direction, self.radius)
        straight_pipe_cross = [first_cross]  # Store all the cross-sections
        straight_pipe_pts = first_cross  # 3 x self.resolution array, holds all points along the cross-section rings of the straight pipe
        l = length / self.resolution  # Distance between cross-section rings
        c1 = np.array(self.end_direction) * l + np.array(self.end_pt)  # Mid point of the second cross-section
        T = np.subtract(c1, np.array(self.end_pt))  # Create translation matrix

        # Build cross-section using points of previous one
        for i in range(1, self.resolution + 1):
            # Translate x, y, and z points of previous cross-section to from the new cross section.
            new_x = straight_pipe_cross[i - 1][0] + T[0]
            new_y = straight_pipe_cross[i - 1][1] + T[1]
            new_z = straight_pipe_cross[i - 1][2] + T[2]
            new_cross = np.array([new_x, new_y, new_z])
            # Update variables.
            straight_pipe_cross.append(new_cross)
            straight_pipe_pts = np.append(straight_pipe_pts, new_cross, axis=1)
        
        # Update self.pts
        if self.pts is not None:
            self.pts = np.append(self.pts, straight_pipe_pts, axis=1)
        else:   
            self.pts = straight_pipe_pts
        
        # Update self.end_pt
        self.end_pt = length * self.end_direction + self.end_pt
        
        # Success message
        print("Straight pipe added successfully, Direction: " + 
              str(self.end_direction) + ", Started from: " + str(prev_start) + 
              ", Ended at: " + str(self.end_pt))
    
    def get_R(_, theta: list[float]) -> list[list[float]]:
        """ Helper function that creates a rotation matrix from angles of rotation.

        Args:
            theta (list[float]):
              Defines angles of rotation in order of x axis, y axis, and z axis.

        Returns:
            list[list[float]]:
              3D rotation matrix of rotation by corresponding angles along
              the x, y, and z axes.
        """
        
        rot_x, rot_y, rot_z = theta
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
        return R_z @ R_y @ R_x 

    def get_axis_angle(_, end: list[float], start: list[float]):
        """ Calculates the axis and angles of rotation to transform the start vector to the end vector.

        Args:
            end (list[float]):
              The vector to transform to.
            start (list[float]):
              The vector to transform.

        Returns:
              First part is the axis, second part is the angle.
        """
        # Normalize the start and end vectors
        start /= np.linalg.norm(start)
        end /= np.linalg.norm(end)
        
        axis = np.cross(start, end) 
        if np.linalg.norm(axis) > 1: 
            axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(start, end))
        
        return axis, angle
    
    def get_quaternion_transform(_, axis: list[float], angle: float) -> list[list[float]]:
        """ Creates a transformation matrix for a rotation about the given axis by the given angle.
        
        Relies on normalized quaternions, then converting that quaternion to a transformation matrix.

        Args:
            axis (list[float]):
              The axis of rotation.
            angle (float):
              The angle of rotation.

        Returns:
            list[list[float]]:
              The transformation matrix.
        """
        
        # Creates a normalized quaternion.
        s = np.sin(angle / 2)
        x = axis[0] * s
        y = axis[1] * s
        z = axis[2] * s
        w = np.cos(angle / 2)
        
        # Converts normalized quaternion to transformation matrix.
        T = t3d.quaternions.quat2mat([w, x, y, z])
        return T
        
    def add_turn_euler(self, theta: list[float]) -> None:
        """ Plots the points for a turn in the pipe by some angles defined by theta.

        Uses the angles to calculate the unit direction vector to turn to, then calls add_turn with it.
        Args:
            theta (list[float]):
              Angles of rotation. First value is angle on x axis, second is angle on y axis, third is angle on z axis.
        """
        
        R = self.get_R(theta)
        v = R @ self.end_direction
        v /= np.linalg.norm(v)
        self.add_turn(v)
    
    def add_turn(self, turn_to: list[float]) -> None:
        """ Plots the points for a turn in the pipe to a specified unit direction vector.

        It first determines the axis and angle of rotation, then places the first
        cross-section at some point. That cross-section is then rotated about that axis
        in increments of that angle. Finally, the turning shape produced is translated to
        connect it to the rest of the pipe.
         
        Args:
            turn_to (list[float]):
              Direction vector to turn to.

        Raises:
            ValueError:
              The turn_to vector is None.
            ValueError:
              The turn_to vector is a list, but does not have 3 values.
            ValueError:
              The calculated rotation angle is 0.
        """
        
        # Checking for exceptions
        if turn_to is None:
            raise ValueError("turn_to is None")
        turn_to = np.array(turn_to)
        if turn_to.size != 3:
            raise ValueError("turn_to must be an of 3 values")                      
        # Find axis and angle of rotation, and check if angle is 0.
        axis1, theta1 = self.get_axis_angle(turn_to, self.end_direction)
        theta_arr = np.linspace(0, theta1, self.curved_res)
        if theta1 == 0:
            raise ValueError("Invalid rotation angle, theta is 0")
        
        # Calculate the point to place the first cross-section. The cross-section
        # should always be facing the same direction as the pipe.
        v = np.cross(axis1, self.end_direction)
        v = -self.turn_radius * v
        turn_arr = []  # Array to hold all the cross-sections in the turning shape.
        cross = self.build_circle(v, self.end_direction, self.radius)
        turn_arr.append(cross)
        end_midpt = v  # Track the center points of each cross-section to get the center point of the last one
        for t in theta_arr:
            T = self.get_quaternion_transform(axis1, t)
            next_cross = T @ cross
            end_midpt = T @ v
            turn_arr.append(next_cross)
            
        # Translate the turning shape to connect it to the rest of the pipe.
        # Determine the translation matrix.
        first_cross_mid = v
        MOVE = [self.end_pt[0] - first_cross_mid[0], self.end_pt[1] - first_cross_mid[1], self.end_pt[2] - first_cross_mid[2]]
        
        # Apply translation matrix on the turning shape.
        n = len(turn_arr)
        for i in range(n):
            x = turn_arr[i][0]
            y = turn_arr[i][1]
            z = turn_arr[i][2]
            new_cross = [x + MOVE[0], y + MOVE[1], z + MOVE[2]]
            turn_arr[i][0] = new_cross[0]
            turn_arr[i][1] = new_cross[1]
            turn_arr[i][2] = new_cross[2]
        
        # Use the translation matrix to find the mid point of the final cross-section.
        end_midpt = np.array(end_midpt) + np.array(MOVE)
        
        # Update class variables.
        for t in turn_arr:
            self.pts = np.append(self.pts, t, axis=1)
        prev_dir = self.end_direction
        self.end_direction = turn_to
        prev_endpt = self.end_pt
        self.end_pt = end_midpt
        # Print success message.
        print(
            "Successful turn, Start direction: " + str(prev_dir) + ",  End direction: " + str(self.end_direction) + 
            ", Started from: " + str(prev_endpt) + ", Ended at: " + str(self.end_pt)
        )   