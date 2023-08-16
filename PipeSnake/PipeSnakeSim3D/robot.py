import numpy as np
from pipe import Pipe
import transforms3d as t3d

""" Builds everything needed for driver.py to plot the pipe snake in a Pipe environment.

The robot uses dimensions of the pipe environment it's placed in to adapt its
leg angles and starting direction. However, if the leg lengths aren't long enough
for the wheels to touch the walls of the pipe, the model is incorrectly built.

TODO:
    - Clean up code, try to remove for loops and vectorize as much as possible
    - Contact sensing implementation, how will it know when it reaches a turn?
    - Animation implementation

Typical usage example:
    p = Pipe(0.2, [0,0,0], [1, 0, 0])
    b = Robot((0, 0, 0), p)
    b.init_bot()
"""

class Robot:
    """ This represents the Pipesnake model.
    
    Note that all float values correspond to meters in real life.
    
    Attributes:
        start_pt:
          A 3-value array to represent the point it starts at.
          The starting point is defined as the point above the front edge of the wheel
          closest to the pipe on the vertical center of the model. See diagram.
        pipe:
          The pipe environment the robot will navigate. Type Pipe from pipe.py.
        wheel_rad:
          The radius of the robot's wheels. Type float.
        wheel_width:
          The width of the robot's wheels. Type float.
        leg_len1:
          The length of the 4 outer legs, see diagram. Type float.
        leg_len2:
          The length of the middle 2 legs, attached to the center rotary joint.
          Type float.
        mid_len:
          The length of the rotary joint. Type float.
        theta1:
          The starting angles for the outer legs. See Diagram. Type float.
        theta2:
          The starting angles for the inner legs. See Diagram. Type float.
        wheels:
          Stores the data necessary for plotting the wheels. Type list[list[float]]
        lines:
          Stores the data necessary for plotting the legs, which
          are just lines. Type list[list[float]]
        resolution:
          Defines how "smooth" the wheels of the bot should look. Type integer.
    """
    
    def __init__(self, start_pt: list[float], pipe: Pipe):
        """Initializes instance of Robot with start_pt and pipe.

        Args:
            start_pt (list[float]):
              The start point of the robot.
            pipe (Pipe):
              The pipe environment the robot will navigate.
        """
        
        self.start_pt = start_pt
        self.pipe = pipe
        self.wheel_rad = 0.05
        self.wheel_width = 0.03
        self.leg_len1 = 0.32
        self.leg_len2 = 0.21
        self.mid_len = 0.21
        self.theta1 = np.arccos((2 * (self.pipe.radius - self.wheel_rad)) / self.leg_len1)
        self.theta2 = np.arccos((self.pipe.radius - self.wheel_rad) / self.leg_len2)
        self.wheels = []                                                                    
        self.lines = []                                                          
        self.resolution = 11
    
    def build_wheel(self, center: list[float], length: float) -> list[list[float]]:
        """ Builds points needed to plot a wheel of the robot.

        Args:
            center (list[float]):
              The center is the center point of the first cross-section of the wheel.
            length (float):
              The length is how long the wheel should be

        Returns:
            list[list[float]]:
              A 2D list of floats for the x, y, and z points of the wheel.
              The first row holds the x values, the second row holds the 
              y values, and the third row holds the z values.
        """
        
        # The way the wheel is plotted is by creating a line of circles as cross-sections, then relying on
        # python's surface plotting to create the outer surface of the pipe from those cross-sections.
        center_x, center_y, center_z = center
        y_pts = np.linspace(center_y, center_y + length, self.resolution)
        theta = np.linspace(0, 2 * np.pi, self.resolution)
        theta_grid, y_grid = np.meshgrid(theta, y_pts)
        x_grid = self.wheel_rad * np.cos(theta_grid) + center_x
        z_grid = self.wheel_rad * np.sin(theta_grid) + center_z

        return [x_grid, y_grid, z_grid]
    
    def init_bot(self) -> None:
        """ Calculates all the data to plot the robot. """
        
        # In the following block, we save some values and functions that will
        # be used repeatedly in the calculations. See diagram.
        dir = self.pipe.end_direction
        center_x, center_y, center_z = self.start_pt
        low = center_z - (self.pipe.radius - self.wheel_rad)  # z value for wheels at bottom of pipe
        hi = low - (2 * self.wheel_rad) + (2 * self.pipe.radius)  # z values for wheels at top of pipe
        y = center_y - (self.wheel_width / 2)  # y values to start building all wheels from
        x_sub1 = (hi - low) * np.tan(self.theta1)  # how much to subtract from x value for wheels on same side
        x_sub2 = (center_z - low) * np.tan(self.theta2)  # how much to subtract from x value to get to left point of rotary joint from left wheel closest to rotary joint
        
        def build_line(start: list[float], end: list[float]) -> list[float]:
            """ Builds an array to represent a line for plotting.
            
            Uses the start point and end point provided to calculate the resulting line segment.
            Also updates the self.lines array

            Args:
                start (list[float]):
                  The start point of the line.
                end (list[float]):
                  The end point of the line.

            Returns:
                list[float]:
                  The end point of the line
            """
            
            line_x = [start[0], end[0]]
            line_y = [start[1], end[1]]
            line_z = [start[2], end[2]]
            line = [line_x, line_y, line_z]
            self.lines.append(line)
            return end
            
        def build_outer_legs(start_pt: list[float]) -> list[float]:
          """ Creates the wheels and lines of the outer legs.

          Args:
              start_pt (list[float]): 
                The point to begin building the leg from. Defined as the the point
                closest to the pipe above the center of the front-most wheel, on the vertical center
                of the model

          Returns:
              list[float]:
                The point over the final wheel built, on the vertical center of the model.
          """

          # The outer legs consist of 3 wheels and 2 lines connecting the centers of the wheels.
          # Two wheels are placed at the bottom of the pipe, and one is between those two but placed
          # at the top of the pipe, forming a triangle. 
          start_x, start_y = start_pt
          midpt0 = [start_x, start_y, low]
          wheel0 = self.build_wheel(midpt0, self.wheel_width)
          self.wheels.append(wheel0)
      
          midpt1 = [midpt0[0] - x_sub1, y, hi]
          wheel1 = self.build_wheel(midpt1, self.wheel_width)
          self.wheels.append(wheel1)
      
          build_line([midpt0[0], center_y, midpt0[2]], [midpt1[0], center_y, midpt1[2]])
          
          midpt2 = [midpt1[0] - x_sub1, y, low]
          wheel2 = self.build_wheel(midpt2, self.wheel_width)
          self.wheels.append(wheel2)
          
          build_line([midpt1[0], center_y, midpt1[2]], [midpt2[0], center_y, midpt2[2]])
          
          return midpt2
        
        # Builds the entire model using previous helper functions.
        # It builds from the section of the model closest to the pipe to the section
        # farthest from the pipe.
        midpt2 = build_outer_legs([center_x - self.wheel_rad, y])  # Build front-most outer legs
        
        # Build the rotary joint and lines connecting the rotary joint to the outer legs.
        end0 = build_line([midpt2[0], center_y, midpt2[2]], [midpt2[0] - x_sub2, center_y, center_z])
        end1 = build_line(end0, [end0[0] - self.leg_len2, end0[1], end0[2]])
        end2 = build_line(end1, [end1[0] - x_sub2, end1[1], low])
        
        # Builds the outer legs farthest from the pipe.
        build_outer_legs([end2[0], y])
        
        # The following code is responsible for orienting the robot in the same direction as the pipe.
        # The following links describe the methodology used for rotating the robot.
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        # https://forums.cgsociety.org/t/rotation-matrix-from-2-vectors/1295254
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        # Calculate the axis and angle of rotation.
        bot_dir = [1, 0, 0]  # Model's initial direction is along the x axis
        dir = self.pipe.start_direction
        v = np.cross(bot_dir, dir)  # Calculate the axis of rotation by cross-producting the directions of the robot and the pipe
        v /= np.linalg.norm(v)
        angle = np.arccos(np.dot(bot_dir, dir))
        
        # Normalize the quaternion, then build the transformation matrix.
        s = np.sin(angle / 2)
        x = v[0] * s
        y = v[1] * s
        z = v[2] * s
        w = np.cos(angle / 2)
        T = t3d.quaternions.quat2mat([w, x, y, z])
        
        # Transform the bot to align with the pipe's direction.
        # Start with aligning the wheels.
        n_wheels = len(self.wheels)
        for i in range(n_wheels):
            x_pts = self.wheels[i][0]
            y_pts = self.wheels[i][1]
            z_pts = self.wheels[i][2]
            rows = len(x_pts)
            cols = len(x_pts[0])
            for r in range(rows):
                for c in range(cols):
                    T_pts = T @ np.array([[x_pts[r][c]], [y_pts[r][c]], [z_pts[r][c]]])
                    self.wheels[i][0][r][c] = T_pts[0]
                    self.wheels[i][1][r][c] = T_pts[1]
                    self.wheels[i][2][r][c] = T_pts[2]
        
        # Rotate the lines
        n_lines = len(self.lines)
        for i in range(n_lines):
            x_pts = self.lines[i][0]
            y_pts = self.lines[i][1]
            z_pts = self.lines[i][2]
            for j in range(2):
                T_pts = T @ np.array([x_pts[j], y_pts[j], z_pts[j]])
                self.lines[i][0][j] = T_pts[0]
                self.lines[i][1][j] = T_pts[1]
                self.lines[i][2][j] = T_pts[2]

        # Print success message
        print("Bot successfully initialized, Start point: " + str(self.start_pt) + ", Direction: " + str(self.pipe.start_direction))