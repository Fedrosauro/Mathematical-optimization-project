import random
import numpy as np
import copy
import pandas as pd

def get_drone_customers(drone_tours):
    drone_customers = []
    for tour in drone_tours:
        for customer in tour:
            drone_customers.append(customer)
    return drone_customers

def get_truck_customers(truck_tours):
    truck_customers = []
    for tour in truck_tours:
        for customer in tour:
            truck_customers.append(customer)
    return truck_customers

def sort_by_angular(coordinates, c_drones, grad_seed):
    rad_seed = np.deg2rad(grad_seed)
    ref_point = coordinates[0]

    def polar_angle(coord):
        dx = coord[0] - ref_point[0]
        dy = coord[1] - ref_point[1]
        angle = np.arctan2(dy, dx)
        return angle

    def angular_distance(angle1, angle2):
        return min(abs(angle1 - angle2), 2 * np.pi - abs(angle1 - angle2))

    angles = [polar_angle(coordinates[i]) for i in c_drones]
    distances = [angular_distance(angle, rad_seed) for angle in angles]
    indexed_distances = list(zip(c_drones, distances))
    indexed_distances.sort(key=lambda x: x[1])
    sorted_c_drones = [index for index, _ in indexed_distances]
    return sorted_c_drones

def sort_by_euclidean_distance(distance_matrix, c_seed):
    distances = [distance_matrix[c_seed][i] for i in range(1, len(distance_matrix[c_seed]))]
    sorted_c = np.argsort(distances)
    sorted_c = [x + 1 for x in sorted_c]
    return sorted_c

def remove_drone_customer(drone_tour, absent_customers, customer_to_remove):
    new_tour = drone_tour
    new_tour.remove(customer_to_remove)
    new_absent_vector = absent_customers + [customer_to_remove]
    return new_tour, new_absent_vector

#this method guarantees that the strings removed aren't longer than the average tour length
def max_string_length(truck_tours, L_max):
    n_non_empty_truck_tours = 0
    for truck_tour in truck_tours:
        if len(truck_tour) > 0:
            n_non_empty_truck_tours += 1
    if n_non_empty_truck_tours == 0:
        return 0
    average_tour_cardinality = len(get_truck_customers(truck_tours)) / (n_non_empty_truck_tours)
    return min(average_tour_cardinality, L_max)

#this method is taken from Slack Induction by String Removals for Vehicle Routing Problems, Christiaens & Berge
def n_strings_to_remove(c_average_removed, l_s_max):
    k_s_max = (4 * c_average_removed) / (1 + l_s_max) - 1
    k_s = int(random.uniform(1, k_s_max + 1))  
    return k_s

def string_to_remove_length(l_s_max, truck_tour):
    l_t_max = min(len(truck_tour), l_s_max)
    l_t = int(random.uniform(1, l_t_max + 1))
    return l_t

def sort_absent_customers(instance, solution, w1, w2, w3, w4, w5):
    absent_customers = solution[1]

    def random_sort(instance, solution, absent_customers):
        random.shuffle(absent_customers)
        return absent_customers
        
    def near_dp(instance, solution, absent_customers):
        absent_customers = sorted(absent_customers, key = lambda customer : instance.distances[0][customer])
        return absent_customers
        
    def far_dp(instance, solution, absent_customers):
        absent_customers = sorted(absent_customers, key = lambda customer : (1/instance.distances[0][customer]))
        return absent_customers
        
    def near_tr(instance, solution, absent_customers):
        truck_customers = get_truck_customers(solution[0][0])
        if (len(truck_customers) == 0):
            return random_sort(instance, solution, absent_customers)
        absent_customers = sorted(absent_customers, key = lambda customer : min(instance.distances[t_customer][customer] for t_customer in truck_customers))
        return absent_customers
        
    def far_tr(instance, solution, absent_customers):
        truck_customers = get_truck_customers(solution[0][0])
        if (len(truck_customers) == 0):
            return random_sort(instance, solution, absent_customers)
        absent_customers = sorted(absent_customers, key = lambda customer : 1/(1+(min(instance.distances[t_customer][customer] for t_customer in truck_customers))))
        return absent_customers

    sorting_methods = [random_sort, near_dp, far_dp, near_tr, far_tr]
    weights = [w1, w2, w3, w4, w5]
    selected_method = random.choices(sorting_methods, weights, k=1)[0]
    return selected_method(instance, solution, absent_customers)

def truck_tour_time(truck_travel_times, tour):   
    if (len(tour) == 0):
        return 0
    total_time = truck_travel_times[0][tour[0]]   
    for i in range(len(tour) - 1):       
        total_time += truck_travel_times[tour[i]][tour[i + 1]]
    total_time += truck_travel_times[tour[-1]][0] 
    return total_time

def drone_tour_time(drone_travel_times, tour):  
    if (len(tour) == 0):
        return 0   
    total_time = 0  
    for i in range(len(tour)):
        total_time += (drone_travel_times[tour[i]]) * 2
    return total_time

def remove_string(truck_tour, string_length, customer, absent_customers):
    customer_index = next((index for index, c in enumerate(truck_tour) if c == customer), -1)
    if (len(truck_tour) < customer_index + string_length):
        absent_customers += truck_tour[len(truck_tour)-string_length:]
        truck_tour = truck_tour[:len(truck_tour)-string_length]
    else:
        absent_customers += truck_tour[customer_index: (customer_index + string_length)]
        truck_tour = truck_tour[:customer_index] + truck_tour[customer_index + string_length:]
    return truck_tour, absent_customers

def is_truck_tour_feasible(instance, tour):
    return (truck_tour_time(instance.t_t, tour) <= instance.T_t) and sum(instance.w[customer] for customer in tour) <= instance.Q_t

def feasible_truck_tour_positions_calculation(instance, truck_tours, customer):
    feasible_positions = []
    for i in range(len(truck_tours)):
            for j in range (len(truck_tours[i]) + 1):
                new_tour = copy.deepcopy(truck_tours[i])
                new_tour.insert(j, customer)
                if (is_truck_tour_feasible(instance, new_tour)):
                     feasible_positions.append([0, i, j])
    return feasible_positions

def is_drone_tour_feasible(instance, tour):
    for customer in tour:
       if (instance.w[customer] > instance.Q_d) or ((instance.t_d[customer] * 2) > instance.d_end):
        return False 
    return (drone_tour_time(instance.t_d, tour) <= instance.T_d)

def is_drone_eligible(instance, drone_tours, customer):
    if (instance.w[customer] > instance.Q_d) or ((instance.t_d[customer] * 2) > instance.d_end):
        return False
    for tour in drone_tours:
        new_tour = copy.deepcopy(tour)
        new_tour.append(customer)
        if (drone_tour_time(instance.t_d, new_tour) <= instance.T_d):
            return True   
    return False

def insert_customer(customer, pos, solution_):    
    solution_[0][pos[0]][pos[1]].insert(pos[2], customer)    
    return solution_

def largest_spatial_slack_drone(instance, drone_tours):
    chosen_drone = min(range(len(drone_tours)), key=lambda i: drone_tour_time(instance.t_d, drone_tours[i]))
    return chosen_drone

def cost(instance, solution):
    total_cost = 0
    for truck_tour in solution[0][0]:
        total_cost += truck_tour_time(instance.t_t, truck_tour) * instance.t_speed * instance.C_T
    for drone_tour in solution[0][1]:
        total_cost += drone_tour_time(instance.t_d, drone_tour) * instance.d_speed * instance.C_D
    return total_cost

#this methods return the n_nearest neighbors based on their euclidean distance from the customer
def select_nearest_neighbors(n_nearest, distances_from_customer):
    indexed_array = [(value, idx) for idx, value in enumerate(distances_from_customer) if idx != 0]    
    indexed_array.sort(key=lambda x: x[0])   
    nearest_neighbors = [idx for value, idx in indexed_array[1:n_nearest + 1]] 
    return nearest_neighbors

def find_customer_in_vehicle_tours(tours, customer):
    for i, row in enumerate(tours):
        for j, element in enumerate(row):
            if element == customer:
                return [i, j]
    return None

def _2_opt_x(instance, solution, customer, neighbor, customer_tour_index):
    neighbor_index = find_customer_in_vehicle_tours(solution[0][0], neighbor)
    customer_index = [customer_tour_index, next((index for index, c in enumerate(solution[0][0][customer_tour_index]) if c == customer), -1)]
    new_tour_1 = solution[0][0][neighbor_index[0]][:neighbor_index[1]] + solution[0][0][customer_index[0]][customer_index[1]:]
    new_tour_2 = solution[0][0][customer_index[0]][:customer_index[1]] + solution[0][0][neighbor_index[0]][neighbor_index[1]:]
    if (is_truck_tour_feasible(instance, new_tour_1) and is_truck_tour_feasible(instance,new_tour_2)):
        solution[0][0][neighbor_index[0]] = new_tour_1
        solution[0][0][customer_index[0]] = new_tour_2
    return solution

def relocate(instance, solution, customer, neighbor, tour_index):
    neighbor_index = next((index for index, c in enumerate(solution[0][0][tour_index]) if c == neighbor), -1)
    proposed_tour = copy.deepcopy(solution[0][0][tour_index])    
    proposed_tour.remove(customer)
    proposed_tour.insert(neighbor_index, customer)
    other_tour = copy.deepcopy(solution[0][0][tour_index])
    other_tour.remove(customer)
    other_tour.insert(neighbor_index + 1, customer)
    if (truck_tour_time(instance.t_t, proposed_tour) > truck_tour_time(instance.t_t, other_tour)):
        proposed_tour = other_tour
    if (is_truck_tour_feasible(instance, proposed_tour)):
        solution[0][0][tour_index] = proposed_tour        
    return solution

def swap(instance, solution, customer, neighbor, tour_index):
    customer_index = next((index for index, c in enumerate(solution[0][0][tour_index]) if c == customer), -1)
    neighbor_index = next((index for index, c in enumerate(solution[0][0][tour_index]) if c == neighbor), -1)
    proposed_tour = copy.deepcopy(solution[0][0][tour_index])
    proposed_tour[customer_index] = neighbor
    proposed_tour[neighbor_index] = customer
    if (is_truck_tour_feasible(instance, proposed_tour)):
        solution[0][0][tour_index] = proposed_tour
    return solution

def _2_opt(instance, solution, customer, neighbor, tour_index):
    customer_index = next((index for index, c in enumerate(solution[0][0][tour_index]) if c == customer), -1)
    neighbor_index = next((index for index, c in enumerate(solution[0][0][tour_index]) if c == neighbor), -1)
    i = min(customer_index, neighbor_index)
    j = max(customer_index, neighbor_index)
    proposed_tour = copy.deepcopy(solution[0][0][tour_index])
    if (i == 0):
        proposed_tour[:j+1] = proposed_tour[j::-1]
        return solution
    proposed_tour[i:j+1] = proposed_tour[j:i-1:-1]
    if (is_truck_tour_feasible(instance, proposed_tour)):
        solution[0][0][tour_index] = proposed_tour
    return solution

def swap_x(instance, solution, customer, neighbor, customer_tour_index):
    customer_index = [customer_tour_index, next((index for index, c in enumerate(solution[0][0][customer_tour_index]) if c == customer), -1)]
    neighbor_index = find_customer_in_vehicle_tours(solution[0][1], neighbor)
    modified_truck_tour = copy.deepcopy(solution[0][0][customer_tour_index])
    modified_truck_tour[customer_index[1]] = neighbor
    modified_drone_tour = copy.deepcopy(solution[0][1][neighbor_index[0]])
    modified_drone_tour[neighbor_index[1]] = customer
    if(is_truck_tour_feasible(instance, modified_truck_tour) and is_drone_tour_feasible(instance, modified_drone_tour)):
        solution[0][0][customer_index[0]] = modified_truck_tour
        solution[0][1][neighbor_index[0]] = modified_drone_tour
    return solution

def shift_t(instance, solution, customer, customer_tour_index):
    if (is_drone_eligible(instance, solution[0][1], customer)):
        solution[0][0][customer_tour_index].remove(customer)
        chosen_drone = largest_spatial_slack_drone(instance, solution[0][1])
        solution[0][1][chosen_drone].append(customer)
    return solution

def shift_d(instance, solution, customer, customer_tour_index):
    new_solution = copy.deepcopy(solution)
    new_solution[0][1][customer_tour_index].remove(customer)
    pos_best = None
    feasible_truck_tour_positions = feasible_truck_tour_positions_calculation(instance, solution[0][0], customer)
    for pos in feasible_truck_tour_positions: 
        if pos_best == None or (pos_best != None and cost(instance, insert_customer(customer, pos, copy.deepcopy(solution))) < cost(instance, insert_customer(customer, pos_best, copy.deepcopy(solution)))):
            pos_best = pos
    if pos_best == None:
        return solution
    solution = insert_customer(customer, pos_best, new_solution)
    return solution

def get_position(customer, tours):
    for vehicle_index in range(len(tours)): 
        for tour_index in range(len(tours[vehicle_index])):
            for customer_index in range(len(tours[vehicle_index][tour_index])):
                if tours[vehicle_index][tour_index][customer_index] == customer:
                    return [vehicle_index, tour_index, customer_index]                
    return None

def is_swap_possible(instance, tours, customer, position):
    new_tour = copy.deepcopy(tours[position[0]][position[1]])
    new_tour[position[2]] = customer
    if (position[0] == 0):
        return is_truck_tour_feasible(instance, new_tour)
    if(position[0] == 1):
        return is_drone_tour_feasible(instance, new_tour)   
    return False

def total_completion_time(instance, solution):
    total_time = 0  
    for truck_tour in solution[0][0]:
        total_time += truck_tour_time(instance.t_t, truck_tour)       
    for drone_tour in solution[0][1]:
        total_time += drone_tour_time(instance.t_d, drone_tour)      
    return total_time

def makespan(instance, solution):
    slowest_tour_time = 0
    for truck_tour in solution[0][0]:
        curr_time_tour = truck_tour_time(instance.t_t, truck_tour)
        if curr_time_tour >= slowest_tour_time:
            slowest_tour_time = curr_time_tour     
    for drone_tour in solution[0][1]:
        curr_time_tour = drone_tour_time(instance.t_d, drone_tour)
        if curr_time_tour >= slowest_tour_time:
            slowest_tour_time = curr_time_tour     
    return slowest_tour_time

def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)