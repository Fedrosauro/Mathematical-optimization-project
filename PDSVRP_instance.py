from scipy.spatial import distance_matrix
import numpy as np
import re

class PDSVRPInstance:
    def __init__(self, file_path):
        values, coordinates, weights = self.load_instance(file_path)

        self.C_T = values["TRUCK UNIT COST"] #truck transportation cost
        self.C_D = values["DRONE UNIT COST"] #drone transportation cost
        self.D = values["NUM DRONES"] #number of drones
        self.N = len(weights) #number of nodes
        self.h = values["NUM TRUCKS"] #number of trucks
        self.distances= self.distance_matrix_computation(coordinates) #matrix of distances
        self.manhattan_distances = self.manhattan_distance_matrix_computation(coordinates)
        self.t_t = self.distances/values["TRUCK SPEED"] #matrix of trucks travel times#########################
        self.t_d =self.distances[0]/values["DRONE SPEED"] #vector of drones travel times
        self.Q_t = values["TRUCK CAP"] #truck capacity
        self.Q_d = values["DRONE CAP"] #drone capacity
        self.T_t = values["TRUCK TIME LIMIT"] #max time truck
        self.T_d = values["DRONE TIME LIMIT"] #max time drones
        self.d_end = values["DRONE ENDURANCE"]

        self.t_speed = values["TRUCK SPEED"]
        self.d_speed = values["DRONE SPEED"]

        self.coordinates = coordinates

        self.w = weights #weights vector
        
    def load_instance(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        # Dictionary of values
        values = {}
    
        # Lists to save coordinates and weights
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
                # Add coordinates and weights to the lists
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
    
    def get_params(self):
        return [self.C_T, self.C_D, self.D, self.N, self.h, self.distances, self.t_t, self.t_d, self.Q_t, self.Q_d, self.T_t, self.T_d, self.w]
