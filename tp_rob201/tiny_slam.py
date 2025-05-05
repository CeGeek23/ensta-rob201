""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid
import matplotlib.pyplot as plt
import heapq
import pickle
import math
from collections import defaultdict

OCCUPANCY_MAX = +1.0
OCCUPANCY_MIN = -1.0

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # On crée une instance d'OccupancyGrid
        self.grid = OccupancyGrid(x_min, x_max, y_min, y_max, resolution)

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        distances = lidar.get_sensor_values() # retourne les distances du lidar
        angles = lidar.get_ray_angles() # angles fournis par le capteur lidar
        
        # array de booleens représentant les obstacles trouvées par le lidar
        is_obstacle = distances < lidar.max_range
        
        # Récupérons la position du robot
        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]
        
        # récupérons les valeurs du lidar avec les obstacles dans le referentiel de l'odométrie
        xObs = x_0 + distances[is_obstacle] * np.cos(angles[is_obstacle] + angle_0)
        yObs = y_0 + distances[is_obstacle] * np.sin(angles[is_obstacle] + angle_0)
        
        # convertissons les coordonnées du lidar dans le referentiel de la map
        xObsMap, yObsMap = self.grid.conv_world_to_map(xObs, yObs)
        
        # gardons seulement les valeurs dans la map
        isValidX = xObsMap < self.grid.x_max_map
        isValidY = yObsMap < self.grid.y_max_map
        
        xObsMap = xObsMap[isValidX * isValidY]
        yObsMap = yObsMap[isValidX * isValidY]
        
        isValidX = 0 <= xObsMap
        isValidY = 0 <= yObsMap
        
        xObsMap = xObsMap[isValidX * isValidY]
        yObsMap = yObsMap[isValidX * isValidY]
        
        # somme des valeurs d'occupation des points d'obstacles
        score = np.sum(self.grid.occupancy_map[xObsMap, yObsMap])
        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TP4

        # initialize reference with not given
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # odometer original reference
        x_odom_ref = odom_pose_ref[0]
        y_odom_ref = odom_pose_ref[1]
        ang_odom_ref = odom_pose_ref[2]

        # robot position in his odometer reference
        x_odom = odom[0]
        y_odom = odom[1]
        ang_odom = odom[2]

        # Calcul plus direct des coordonnées corrigées
        cos_ref = np.cos(ang_odom_ref)
        sin_ref = np.sin(ang_odom_ref)
        
        # Rotation et translation pour obtenir les coordonnées dans le référentiel global
        x_corrected = x_odom_ref + x_odom * cos_ref - y_odom * sin_ref
        y_corrected = y_odom_ref + x_odom * sin_ref + y_odom * cos_ref
        ang_corrected = ang_odom + ang_odom_ref
        
        # Normaliser l'angle entre -pi et pi
        ang_corrected = np.arctan2(np.sin(ang_corrected), np.cos(ang_corrected))
        
        return [x_corrected, y_corrected, ang_corrected]

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # on initialise le score avec la position de référence
        best_score = self._score(lidar, raw_odom_pose)
        best_ref = self.odom_pose_ref
        
        # paramètres de la recherche (variations aléatoires)
        i = 0
        N = 1.5e2
        
        # valeur aléatoire suivant une distribution gaussienne
        while i < N:
            # offset aléatoire
            offset = []
            sigma = np.array([5.0, 5.0, 0.15]) # ecart-types de la distribution gaussienne multidimensionnelle
            
            offset.append(np.random.normal(0.0, sigma[0]))
            offset.append(np.random.normal(0.0, sigma[1]))
            offset.append(np.random.normal(0.0, sigma[2]))
            
            # on ajoute l'offset à la position de référence de l'odométrie
            new_ref = best_ref + offset
            
            # ajout de l'offset à la référence
            odom_offset = self.get_corrected_pose(raw_odom_pose, new_ref)
            offset_score = self._score(lidar, odom_offset)
            
            # si un nouveau score est trouvé, on réinitialise le nombre d'essais
            if offset_score >= best_score:
                i = 0
                best_score = offset_score
                best_ref = new_ref
            # on continue la recherche
            else:
                i += 1
        # on sauvegarde la meilleure référence trouvée
        self.odom_pose_ref = best_ref
        print(best_score)
        
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # * TP3
        # get lidar values, robot referencial
        distances = lidar.get_sensor_values()   # 
        angles = lidar.get_ray_angles()         # angles in radiums

        # define threshold for border values
        border = 20

        # get array of bool's where the lidar found _obstacles
        is_obstacle = distances <= (lidar.max_range - border)

        # get robot's position, odometer referencial (robot's initial position)
        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]

        # get lidar values, odometer referencial
        x = distances * np.cos(angles + angle_0) + x_0
        y = distances * np.sin(angles + angle_0) + y_0

        # increase points values, _obstacle
        self.grid.add_map_points(x[is_obstacle], y[is_obstacle], +0.35)  # modèle simple

        # decrease points values, free path
        for x_i, y_i in zip(x, y):
            self.grid.add_value_along_line(x_0, y_0, x_i, y_i, -0.10)

        # set upper and lower limit of point's value
        self.grid.occupancy_map[self.grid.occupancy_map >= OCCUPANCY_MAX] = OCCUPANCY_MAX
        self.grid.occupancy_map[self.grid.occupancy_map <= OCCUPANCY_MIN] = OCCUPANCY_MIN

    def read_map(self, map_csv: str) -> None:
        """
        read Bayesian map from csv file
        map_csv : path to saved map as csv
        """

        try:
            self.grid.occupancy_map = np.genfromtxt(map_csv, delimiter=',')
        except IOError:
            print(f'error: file {map_csv} not found')

        return None


