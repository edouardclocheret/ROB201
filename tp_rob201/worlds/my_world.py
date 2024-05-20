import math
import random
from typing import Type, Union

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.simu_world.closed_playground import ClosedPlayground
from place_bot.simu_world.world_abstract import WorldAbstract

from worlds import walls_my_world


class MyWorld(WorldAbstract):

    def __init__(self, robot: RobotAbstract):
        super().__init__(robot=robot)

        # PARAMETERS MAP
        self._size_area = (1113, 750)

        # PLAYGROUND
        self._playground = ClosedPlayground(size=self._size_area)
        walls_my_world.add_walls(self._playground)
        walls_my_world.add_boxes(self._playground)

        # POSITION OF THE ROBOT
        #angle = random.uniform(-math.pi, math.pi) # math.pi/2.0
        angle =  - math.pi/2.0 
        self._robot_pos = ((439.0, 195), angle)
        self._playground.add(robot, self._robot_pos)

