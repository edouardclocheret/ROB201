""" A set of robotics control functions """

import random
import numpy as np
from math import atan2

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    
    if min(lidar.get_sensor_values()[178:183])<30 :
        rotation_speed = 1.0
        speed = 0.0
    else :
        rotation_speed = 0.0
        speed = 0.3

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command



def potential_field_control(lidar, current_pose, goal_pose, my_robot):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    
    #Composante du gradient qui attire vers l'objectif
    distance_to_goal = np.linalg.norm (current_pose[0:2]-goal_pose[0:2])
    K_goal = 1 #une valeur pas trop élevée est nécessaire pour avoir le temps de tourner
    
    d_quad = 50 #Passage à un potentiel quadratique près de l'objectif

    if distance_to_goal < d_quad :
        #Potentiel attractif quadratique vers le goal (norme qui diminiue)
        gradient_goal = K_goal/d_quad * (goal_pose[0:2] - current_pose[0:2])
    else :
        #Potentiel attractif linéaire vers le goal (norme constante)
        gradient_goal = K_goal/distance_to_goal * (goal_pose[0:2] - current_pose[0:2])


    #Composante du gradinet qui repousse des obstacles
    distance_to_obstacle = np.min(lidar.get_sensor_values())
    angle_to_obstacle = lidar.get_ray_angles()[np.argmin(lidar.get_sensor_values())]
    obs_pose = current_pose[0:2] + distance_to_obstacle*np.array([np.cos(angle_to_obstacle),np.sin(angle_to_obstacle)])

    K_obs = 500000

    d_safe = 350 #Un obstacle n'a pas d'influence à plus de d_safe
    if distance_to_obstacle > d_safe :
        gradient_obstacle = [0,0]
    else :
        gradient_obstacle = K_obs/ (distance_to_obstacle)**3 *(1/distance_to_obstacle - 1/d_safe) * (obs_pose - current_pose[0:2])

    #Combinaison de la partie du gradient provennant du goal et de l'obstacle
    gradient = gradient_goal - gradient_obstacle
    
    #Commande associée au gradient :
    d_arret = 20 #arrêt lorsque l'objectif est atteint
    d_crash = 10

    if distance_to_goal < d_arret : 
        print("Goal reached")
        forward = 0
        rotation = 0
        my_robot.change_goal_state()


    elif distance_to_obstacle < d_crash :
        forward =0
        if angle_to_obstacle <0 :
            rotation = 0.7
        else :
            rotation = -0.7

    else :

        k_forward = 0.2
        k_rotation = 0.6
        
        forward = k_forward
        rotation =  k_rotation *  (atan2(gradient[1], gradient[0]) - current_pose[2])
        
        #empeche de prendre les valeurs interdites
        rotation = max(-0.4,min(0.4,rotation))
        forward = max(-0.7,min(0.7,forward))
    

    command = {"forward": forward,
               "rotation": rotation}
    #print(command)


    return command
