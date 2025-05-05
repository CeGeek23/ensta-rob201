"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, wall_follow, potential_attraction
from occupancy_grid import OccupancyGrid
from planner import Planner
import datetime

SCORE_MIN = +50

# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""


    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        # array of calculates scores used for debug
        self.array_score = []

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min= -self._size_area[0],
                                  x_max= +self._size_area[0],
                                  y_min= -self._size_area[1],
                                  y_max= +self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.command_choice = 'wall_follow'
        self.save_map = False
        self.explore = True
        self.explore_counter_limit = 1000

        self.path = None
        self.path_return = None

    # Donne les commandes au robot en fonction des données capteurs
    def control(self):
        """
        Main control function executed at each time step
        """

        # ! setup
        # default command
        command = {"forward": 0, "rotation": 0}

        # ! exploration
        if self.explore is True:
            # explore map to create a cartography
            if self.counter < self.explore_counter_limit:

                # * initialize occupancy map
                if self.counter <= 50:
                    # use robot odometer without any correction
                    self.tiny_slam.update_map(self.lidar(), self.odometer_values())

                else:
                    # search best reference correction
                    score = self.tiny_slam.localise(self.lidar(), self.odometer_values())

                    # update occupancy map only with corrected reference is good enough
                    if score > SCORE_MIN:
                        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

                # * command choice
                if self.counter > 75:

                    if self.command_choice == 'reactive':
                        command = reactive_obst_avoid(self.lidar())

                    elif self.command_choice == 'potential_field':
                        command = potential_field_control(self.lidar(), self.odometer_values(), np.array([-100.0, -400.0, 0]))

                    elif self.command_choice == 'wall_follow':
                        command = wall_follow(self.lidar())

                    else:
                        command = {"forward": 0, "rotation": 0}

        # ! return
        # until arrive at the goal
        if self.path_return:

            # get current position
            x_pos, y_pos, t_pos = self.tiny_slam.get_corrected_pose(self.odometer_values())
            position = (x_pos, y_pos, t_pos)

            # get next goal
            x_map, y_map = self.path_return[0]
            x_goal, y_goal = self.tiny_slam._conv_map_to_world(x_map, y_map)
            goal = (x_goal, y_goal)

            distance_goal = np.sqrt((x_pos - x_goal)**2 + (y_pos - y_goal)**2)


            # if is already close enough, remove closest nodes
            if distance_goal < 15:
                self.path_return = self.path_return[10:]

            command = potential_attraction(np.array(position), goal)


        # ! update
        # display occupancy map within a certain frequency
        if self.counter % 1 == 0:
            self.tiny_slam.display2(self.odometer_values(), self.path)

        # ! increase counter
        self.counter += 1
        # print(f'{self.counter}')

        return command

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        #self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar(), self.odometer_values())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-50,-50,0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command
