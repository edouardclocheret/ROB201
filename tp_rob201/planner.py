import numpy as np

from occupancy_grid import OccupancyGrid

import heapq

class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_free_neighbors(self,current_cell):
        """for i in range(500):
            current_cell = [i, 400]
            print("observed cell",current_cell , "value", self.grid.occupancy_map[current_cell[0]][current_cell[1]])"""
        result = []
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                if i !=0 or j!=0 :
                    observed_cell = current_cell+np.array([i,j])
                    #print("test de ", self.grid.occupancy_map[observed_cell[0]][observed_cell[1]], "<0.0")
                    
                    bad_environment =False
                    for a in range(-1,2,1):
                        for b in range(-1,2,1):
                            if self.grid.occupancy_map[observed_cell[0]+a][observed_cell[1]+b] >= 1.0:
                                bad_environment =True
                    
                    if not bad_environment:
                        result.append(observed_cell)
                    else:
                        print("neigbor rejected", observed_cell, "value",self.grid.occupancy_map[observed_cell[0]][observed_cell[1]])

        if len(result)==0 :
            print("pas de voisins !")
            print(np.unique(self.grid.occupancy_map))
            exit(0)
            
        return result

    def heuristic (cell_1 , cell_2):
        cell_1 = np.array(cell_1)
        cell_2 = np.array(cell_2)
        return np.linalg.norm(cell_1 -cell_2, 2)

    def manhattan_distance(cell_1, cell_2):

        return np.abs(cell_1[0]-cell_2[0])+ np.abs(cell_1[1]- cell_2[1])

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        #TP5
        start_map = OccupancyGrid.conv_world_to_map(self.grid,start[0],start[1])
        goal_map = OccupancyGrid.conv_world_to_map(self.grid,goal[0], goal[1])
        
        
        path_map = Planner.A_star(self,start_map, goal_map, Planner.heuristic, Planner.get_free_neighbors, Planner.manhattan_distance)     
        
        path_world = []
        for i in range(len(path_map)):
            path_world.append(OccupancyGrid.conv_map_to_world(self.grid,path_map[i][0],path_map[i][1]))

        
        return np.array(path_world)
    
    def reconstruct_path(came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        return total_path
    
    def A_star(self,start, goal, h, neighbor_function, distance):
        """
        A* algorithm for finding the shortest path from start to goal.
        """
        
        open_set = [(h(start, goal), start)]  # Priority queue (min heap) avec les noeuds à évaluer
        came_from = {}  # Path qui se construit petit à petit
        g_score = {start: 0}  # Liste des couts

        print("Processing A*, please wait")
        while open_set:
            
            _, current = heapq.heappop(open_set) #Noeud avec le plus petit f = g + heuristique
            if current == goal:
                return Planner.reconstruct_path(came_from, current) 
            for neighbor in neighbor_function(self,current):
                tentative_g_score = g_score[current] + distance(current, neighbor)
                
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score  
                    f_score = tentative_g_score + h(neighbor_tuple, goal)
                    heapq.heappush(open_set, (f_score, neighbor_tuple))  
                    
        
        return None  #si pas de chemin trouvé

    


    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
