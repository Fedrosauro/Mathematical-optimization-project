import random
import copy
import utilities as u

# R-
##########################################################################################
def sweep_removal_operator(instance, solution, sigma):
    c_drones = u.get_drone_customers(solution[0][1])
    nbSweept = random.randint(0,int(len(c_drones) * sigma))

    grad_seed = random.randint(0, 359)
    c_drones = u.sort_by_angular(instance.coordinates, c_drones, grad_seed)
    for c in c_drones:
        if (len(solution[1]) < nbSweept):

            tour_index = next((index for index, drone_tour in enumerate(solution[0][1]) if c in drone_tour), -1)
            solution[0][1][tour_index], solution[1] = u.remove_drone_customer(solution[0][1][tour_index], solution[1], c)
                               
    return solution

def random_drone_customer_removal(solution, sigma):

    c_drones = u.get_drone_customers(solution[0][1])
    q = random.randint(0,int(len(c_drones) * sigma))
    
    while (len(solution[1]) < q):
        customer_to_remove = random.choice(c_drones)

        c_drones.remove(customer_to_remove)
        tour_index = next((index for index, drone_tour in enumerate(solution[0][1]) if customer_to_remove in drone_tour), -1)
        solution[0][1][tour_index], solution[1] = u.remove_drone_customer(solution[0][1][tour_index], solution[1], customer_to_remove)

    return solution

def string_removal(instance, solution, c_average_removed, L_max): 
    l_s_max = u.max_string_length(solution[0][0], L_max)
    k_s = u.n_strings_to_remove(c_average_removed, l_s_max)
    
    c_seed = random.randint(1, instance.N - 1)
    R = []
    c_adj = u.sort_by_euclidean_distance(instance.distances, c_seed)

    for c in c_adj:
        if len(R) < k_s:
            if any(c in truck_tour for truck_tour in solution[0][0]):
                if c not in solution[1]:
                    tour_index = next((index for index, truck_tour in enumerate(solution[0][0]) if c in truck_tour), -1)
                    if tour_index not in R:
                        l = u.string_to_remove_length(l_s_max, solution[0][0][tour_index])
                        
                        solution[0][0][tour_index], solution[1] = u.remove_string(solution[0][0][tour_index], l, c, solution[1])
                        R.append(tour_index)
            elif any(c in drone_tour for drone_tour in solution[0][1]):
                if c not in solution[1]:
                    tour_index = next((index for index, drone_tour in enumerate(solution[0][1]) if c in drone_tour), -1)
                    solution[0][1][tour_index], solution[1] = u.remove_drone_customer(solution[0][1][tour_index], solution[1], c)

    return solution
##########################################################################################

# R+
def recreate(instance, solution, w1, w2, w3, w4, w5, gamma):
    solution[1] = u.sort_absent_customers(instance, solution, w1, w2, w3, w4, w5)
    while len(solution[1]) > 0:
        c = solution[1][0]
        pos_best = None
        
        feasible_truck_tour_positions = u.feasible_truck_tour_positions_calculation(instance, solution[0][0], c)
       
        for pos in feasible_truck_tour_positions: 
            if pos_best == None or (pos_best != None and u.cost(instance, u.insert_customer(c, pos, copy.deepcopy(solution))) < u.cost(instance, u.insert_customer(c, pos_best, copy.deepcopy(solution))) and random.random() > (1 - gamma)):
                pos_best = pos
        
        if u.is_drone_eligible(instance, solution[0][1], c):
            chosen_drone = u.largest_spatial_slack_drone(instance, solution[0][1])
            if pos_best == None or (pos_best != None and u.cost(instance, u.insert_customer(c, [1, chosen_drone, 0], copy.deepcopy(solution))) < u.cost(instance, u.insert_customer(c, pos_best, copy.deepcopy(solution))) and random.random() > (1 - gamma)):
                pos_best = [1, chosen_drone, 0]

        if pos_best == None:
            empty_truck_tour_index = next((index for index, truck_tour in enumerate(solution[0][0]) if len(truck_tour) == 0), -1)
            if empty_truck_tour_index == -1:
                solution[0][0].append([])
                empty_truck_tour_index = len(solution[0][0]) - 1
            
            pos_best = [0, empty_truck_tour_index, 0]
        
        solution = u.insert_customer(c, pos_best, solution)
        solution[1].remove(c)

    return solution

# R- & R+
def ruin_and_recreate(instance, solution, sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma):
    r = random.random()
    if (r < 0.5):
        solution = sweep_removal_operator(instance, solution, sigma)
    else:
        solution = random_drone_customer_removal(solution, sigma)

    solution = string_removal(instance, solution, c_average_removed, L_max)

    solution = recreate(instance, solution, w1, w2, w3, w4, w5, gamma)

    return solution

# local search is recoursively performed until no improvements are found
def local_search(instance, solution, n_nearest):
    for i in range(len(solution[0][0])):
        
        for customer in solution[0][0][i]:
        
            neighbors = u.select_nearest_neighbors(n_nearest, instance.distances[customer])
            for neighbor in neighbors:
                if (neighbor in u.get_truck_customers(solution[0][0])):
                    neighbor_index = next((j for j, node in enumerate(solution[0][0][i]) if node == neighbor), None)
                    if neighbor_index == None:
                        new_solution = u._2_opt_x(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.cost(instance, new_solution) < u.cost(instance, solution):
                            return local_search(instance, new_solution, n_nearest)
                            
                    else: 
                        new_solution = u.relocate(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.cost(instance, new_solution) < u.cost(instance, solution):
                            return local_search(instance, new_solution, n_nearest)
                            
                        new_solution = u.swap(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.cost(instance, new_solution) < u.cost(instance, solution):
                            return local_search(instance, new_solution, n_nearest)                     

                        new_solution = u._2_opt(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.cost(instance, new_solution) < u.cost(instance, solution):
                            return local_search(instance, new_solution, n_nearest)
                            
                else:
                    new_solution = u.swap_x(instance, copy.deepcopy(solution), customer, neighbor, i)
                    if u.cost(instance, new_solution) < u.cost(instance, solution):
                        return local_search(instance, new_solution, n_nearest)                  

            new_solution = u.shift_t(instance, copy.deepcopy(solution), customer, i)
            if u.cost(instance, new_solution) < u.cost(instance, solution):
                return local_search(instance, new_solution, n_nearest)
            
    for i in range(len(solution[0][1])):
       
        for customer in solution[0][1][i]:
            new_solution = u.shift_d(instance, copy.deepcopy(solution), customer, i)
            if u.cost(instance, new_solution) < u.cost(instance, solution):
                return local_search(instance, new_solution, n_nearest)      

    return solution
            
def initial_solution_construction(instance, w1, w2, w3, w4, w5, gamma, n_nearest):
    A = [c for c in range (1, instance.N)]
    solution = recreate(instance, [[[[] for _ in range(instance.h)], [[] for _ in range(instance.D)]],A], w1, w2, w3, w4, w5, gamma)
    solution = local_search(instance, solution, n_nearest)
    return solution

def perturbate(instance, solution, p_min, p_max, max_unfeasible_swaps):
    p = random.randint(p_min, p_max)
    swaps_executed = 0
    unfeasible_swaps = 0
    while (swaps_executed < p and unfeasible_swaps < max_unfeasible_swaps):
        c1 = random.randint(1, instance.N - 1)
        c2 = random.randint(1, instance.N - 1)
        pos1 = u.get_position(c1, solution[0])
        pos2 = u.get_position(c2, solution[0])
        if (u.is_swap_possible(instance, solution[0], c1, pos2) and u.is_swap_possible(instance, solution[0], c2, pos1)):
            solution[0][pos1[0]][pos1[1]][pos1[2]] = c2
            solution[0][pos2[0]][pos2[1]][pos2[2]] = c1
            swaps_executed += 1
            unfeasible_swaps = 0
        else:
            unfeasible_swaps += 1

    return solution

def SISSRs(instance, sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma, n_nearest, delta, epsilon, iter_imp, iter_max, p_min, p_max, max_unfeasible_swaps_perturb):
    s_0 = initial_solution_construction(instance, w1, w2, w3, w4, w5, gamma, n_nearest)
    s_curr = s_0
    s_best = s_0
    iterations_without_improvement = 0
    iteration_counter = 0
    while (iteration_counter < iter_max):
        s = ruin_and_recreate(instance, copy.deepcopy(s_curr), sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma)
        if u.cost(instance, s) < u.cost(instance, s_curr)*(1+delta):
            s_curr = local_search(instance, s, n_nearest)
            if u.cost(instance, s_curr) < u.cost(instance, s_best):
                s_best = s_curr
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        if iterations_without_improvement >= iter_imp:
            s_curr = perturbate(instance, s_curr, p_min, p_max, max_unfeasible_swaps_perturb)
            iterations_without_improvement = 0
        delta = delta * epsilon
        iteration_counter+=1

    return s_best