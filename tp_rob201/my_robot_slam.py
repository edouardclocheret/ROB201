"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner

from math import pi

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

        # Init SLAM object
        self._size_area = (800, 800)
        self.occupancy_grid = OccupancyGrid(x_min=- self._size_area[0],
                                            x_max=self._size_area[0],
                                            y_min=- self._size_area[1],
                                            y_max=self._size_area[1],
                                            resolution=2)
        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.start = [439.0, 195,0]
        self.goal = [50,-500,0] 
        self.goalReached = False
        self.trajet_printed = False

    def change_goal_state(self):
        self.goalReached = True



    

    def control(self):
        """
        Main control function executed at each time step
        """
        
        if  self.goalReached :

            
            #L'arrivée est le départ et inversement
            traj = Planner.plan(self.planner,np.array(self.start[0:2])+np.array(self.goal[0:2]), np.array(self.start[0:2]))
            

            
            self.occupancy_grid.display_cv(self.odometer_values(), self.goal,traj)
            
            if not self.trajet_printed :
                print(traj,"^^^^\nCeci est le trajet retour obtenu par A*")
                self.trajet_printed =True

            return {"forward": 0.0, "rotation": 0.0}
        
        else :
            #TP3 :
            self.tiny_slam.update_map(self.lidar(),self.odometer_values())
            #TP4 :
            self.tiny_slam.localise(self.lidar(),self.odometer_values())

            return self.control_tp2(self.goal)

    def control_tp1(self):
        """
        Control function for TP1
        """
        #Cette fonction ne sert à rien (exercice pour utiliser le profiler)
        #self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self, goal):
        """
        Control function for TP2
        """
        pose = self.odometer_values()
        
        

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal, self)

        return command
