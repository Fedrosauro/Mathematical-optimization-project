from scipy.spatial import distance_matrix
import numpy as np
import re

class instancePDSVRP:
    def __init__(self, file_path):
        values, coordinates, weights = self.load_instance(file_path)

        self.C_T = values["TRUCK UNIT COST"]
        self.C_D = values["DRONE UNIT COST"]
        self.D = values["NUM DRONES"]
        self.N = len(weights) #number of nodes
        self.h = values["NUM TRUCKS"]
        self.distances= self.distance_matrix_computation(coordinates) #matrix of euclidean distances
        self.manhattan_distances = self.manhattan_distance_matrix_computation(coordinates) #matrix of manhattan distances
        self.t_t = (self.manhattan_distances)/(values["TRUCK SPEED"]) #matrix of trucks travel times
        self.t_d = self.distances[0]/values["DRONE SPEED"] #vector of drones travel times
        self.Q_t = values["TRUCK CAP"] #truck capacity
        self.Q_d = values["DRONE CAP"] #drone capacity
        self.T_t = values["TRUCK TIME LIMIT"]
        self.T_d = values["DRONE TIME LIMIT"]
        self.d_end = values["DRONE ENDURANCE"]
        self.t_speed = values["TRUCK SPEED"]
        self.d_speed = values["DRONE SPEED"]
        self.coordinates = coordinates
        self.w = weights
        
    def load_instance(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        values = {}
    
        coordinates = []
        weights = []

        for line in lines:
            if ',' in line:
                key, value = line.strip().split(',')
                if key in ["NUM DRONES", "NUM TRUCKS"]:
                    values[key] = int(value)
                elif key in ["TRUCK CAP", "DRONE CAP", "TRUCK SPEED", "DRONE SPEED", "DRONE ENDURANCE", "DRONE TIME LIMIT", "TRUCK TIME LIMIT", "TRUCK UNIT COST", "DRONE UNIT COST"]:
                    values[key] = float(value)
            else:
                parts = re.sub(' +',' ', line).split()[1:] 
                x, y, weight = map(float, parts)
                coordinates.append((x, y))
                weights.append(weight)

        return values, coordinates, weights
    
    def distance_matrix_computation(self, coord):
        coord_array = np.array(coord)
        return distance_matrix(coord_array, coord_array)
    
    def manhattan_distance_matrix_computation(self, coord):
        coord_array = np.array(coord)
        num_points = coord_array.shape[0]
        manhattan_dist_matrix = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            for j in range(num_points):
                manhattan_dist_matrix[i, j] = np.sum(np.abs(coord_array[i] - coord_array[j]))
        
        return manhattan_dist_matrix
