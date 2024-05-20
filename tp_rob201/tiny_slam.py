""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        
        #for the engineer to finetune the localization
        self.counter_ameliorations =0
    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        radiuses = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        x_world = pose[0] + radiuses * np.cos(angles + pose[2])
        y_world = pose[1] + radiuses * np.sin(angles + pose[2])
        x_map, y_map = self.grid.conv_world_to_map(x_world,y_world)
        
        #there are a lot of points that should not be counted :
        map = self.grid.occupancy_map

        #deleting the values where the maximal distance of the laser is reached
        mask_1 = radiuses < 550
        x_map = x_map[mask_1]
        y_map = y_map[mask_1]

        xx,yy = 800,800
        #deleting the values where the indexes are out of the map
        mask = (x_map > 0) & (x_map < xx) & (y_map > 0) & (y_map < yy)
        
        score = np.sum(map[x_map[mask], y_map[mask]])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        #TODO for TP4
        if odom_pose_ref is None : 
            odom_pose_ref = self.odom_pose_ref
        
        d0= np.linalg.norm(odom_pose[0:2])
        a0= np.arctan2(odom_pose[1], odom_pose[0])
        corrected_pose = np.array([odom_pose_ref[0] + d0 * np.cos(odom_pose_ref[2] + a0),
                                    odom_pose_ref[1]+d0 * np.sin(odom_pose_ref[1]+a0),
                                    odom_pose_ref[2]+odom_pose[2]])
        
        

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        raw_odom_pose : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        variance_pos = 0.0 #en x et y
        variance_theta = 0.01

        best_score = self._score(lidar, raw_odom_pose)
        best_ref_pose = self.odom_pose_ref

        N = 200  # nombre de tirages de bruits
        for i in range (N):
            variance_pos = 0.1 #en x et y
            variance_theta = 0.01
            # Generate a random offset in 3 dimensions
            offset = np.random.multivariate_normal(mean=np.zeros(3), cov=np.diag([variance_pos, variance_pos, variance_theta]))
            #print("offest",offset)
            new_ref_pose = best_ref_pose + offset
            new_odom_pose = self.get_corrected_pose(raw_odom_pose,new_ref_pose)
            # Calculate the score with the updated pose
            new_score = self._score(lidar, new_odom_pose)
            #print("new score",new_score)

            # Check if the new score is better
            if new_score > best_score:
                #print("improving position")
                best_score = new_score
                best_ref_pose =new_ref_pose
                self.counter_ameliorations +=1
            #print ("best score",best_score)
        
        if best_score > 0 :
            self.odom_pose_ref = best_ref_pose
            
        
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        radiuses = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        x_obs = pose[0] + radiuses * np.cos(angles + pose[2])
        y_obs = pose[1] + radiuses * np.sin(angles + pose[2])
        """x_1 = pose[0] + (radiuses-3)* np.cos(angles + pose[2])
        y_1 = pose[1] + (radiuses-3) * np.sin(angles + pose[2])"""
        x_2 = pose[0] + (radiuses-30) * np.cos(angles + pose[2])
        y_2 = pose[1] + (radiuses-30) * np.sin(angles + pose[2])
        
        #positions des obstacles détectés par le lidar
        
        for i in range(360):
            #ajout de probas faibles le long du rayon lidar
            self.grid.add_map_line( pose[0], pose[1], x_2[i], y_2[i], -2)
            
        
            """if radiuses[i]<550 :
                #Ajout d'une zone de proba 1/2 avant les murs
                self.grid.add_map_line(x_1[i], y_1[i], x_2[i], y_2[i], 0 )"""
            #ajoiter 0 revient à ne rien faire

        #Si la portée maximale du lidar est atteinte, 
        #on n'ajoute pas de mur
        masque = radiuses<550
        #ajout d'une proba élevée à la position de l'obstacle
        self.grid.add_map_points(x_obs[masque], y_obs[masque], 2 )

        #Seuillage
        self.grid.occupancy_map[self.grid.occupancy_map < -5] = -5
        self.grid.occupancy_map[self.grid.occupancy_map > 5] = 5
        
        self.grid.display_cv(pose)        

    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])



