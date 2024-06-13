import random
import copy
import utilities as u
import heuristic as h

#First modification implemented: best position computation
def recreate_min_time(instance, solution, w1, w2, w3, w4, w5, gamma):
    solution[1] = u.sort_absent_customers(instance, solution, w1, w2, w3, w4, w5)
    while len(solution[1]) > 0:
        c = solution[1][0]
        pos_best = None
        makespan_best = float('inf')
        
        feasible_truck_tour_positions = u.feasible_truck_tour_positions_calculation(instance, solution[0][0], c)

        for pos in feasible_truck_tour_positions: 
            makespan_diff = u.truck_tour_time(instance.t_t, u.insert_customer(c, pos, copy.deepcopy(solution))[0][pos[0]][pos[1]]) - u.makespan(instance, copy.deepcopy(solution))
            if pos_best == None or (pos_best != None and makespan_diff < makespan_best and random.random() > (1 - gamma)):
                makespan_best = makespan_diff
                pos_best = pos
        
        if u.is_drone_eligible(instance, solution[0][1], c):
            chosen_drone = u.largest_spatial_slack_drone(instance, solution[0][1]) 
            makespan_diff = u.drone_tour_time(instance.t_d, u.insert_customer(c, [1, chosen_drone, 0], copy.deepcopy(solution))[0][1][chosen_drone]) - u.makespan(instance, copy.deepcopy(solution))
            if pos_best == None or (pos_best != None and makespan_diff < makespan_best and random.random() > (1 - gamma)):
                makespan_best = makespan_diff
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

def ruin_and_recreate_min_time(instance, solution, sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma):
    r = random.random()
    if (r < 0.5):
        solution = h.sweep_removal_operator(instance, solution, sigma)
    else:
        solution = h.random_drone_customer_removal(solution, sigma)

    solution = h.string_removal(instance, solution, c_average_removed, L_max)

    solution = recreate_min_time(instance, solution, w1, w2, w3, w4, w5, gamma)

    return solution

#Second modification implemented: local improvement evaluation
def local_search_min_time(instance, solution, n_nearest):

    for i in range(len(solution[0][0])):

        for customer in solution[0][0][i]:

            neighbors = u.select_nearest_neighbors(n_nearest, instance.distances[customer])
            for neighbor in neighbors:
                neighbor_pos = u.get_position(neighbor, copy.deepcopy(solution[0]))

                if (neighbor in u.get_truck_customers(solution[0][0])):
                    neighbor_index = next((j for j, node in enumerate(solution[0][0][i]) if node == neighbor), None)
                    
                    if neighbor_index == None:
                        t_1 = u.truck_tour_time(instance.t_t, copy.deepcopy(solution[0][0][i]))
                        t_2 = u.truck_tour_time(instance.t_t, copy.deepcopy(solution[0][neighbor_pos[0]][neighbor_pos[1]]))
                        t = max(t_1, t_2)

                        new_solution = u._2_opt_x(instance, copy.deepcopy(solution), customer, neighbor, i)
                        t_1_x = u.truck_tour_time(instance.t_t, copy.deepcopy(new_solution[0][0][i]))
                        t_2_x = u.truck_tour_time(instance.t_t, copy.deepcopy(new_solution[0][neighbor_pos[0]][neighbor_pos[1]]))
                        
                        if ((t_1_x < t and t_2_x < t) or (t_1_x < t and t_2_x < t)):
                            return local_search_min_time(instance, new_solution, n_nearest)

                    else: 
                        new_solution = u.relocate(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.total_completion_time(instance, copy.deepcopy(new_solution)) < u.total_completion_time(instance, copy.deepcopy(solution)):
                            return local_search_min_time(instance, new_solution, n_nearest)

                        new_solution = u.swap(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.total_completion_time(instance, copy.deepcopy(new_solution)) < u.total_completion_time(instance, copy.deepcopy(solution)):
                            return local_search_min_time(instance, new_solution, n_nearest)

                        new_solution = u._2_opt(instance, copy.deepcopy(solution), customer, neighbor, i)
                        if u.total_completion_time(instance, copy.deepcopy(new_solution)) < u.total_completion_time(instance, copy.deepcopy(solution)):
                            return local_search_min_time(instance, new_solution, n_nearest)

                else:
                    t_1 = u.truck_tour_time(instance.t_t, copy.deepcopy(solution[0][0][i]))
                    t_2 = u.drone_tour_time(instance.t_d, copy.deepcopy(solution[0][neighbor_pos[0]][neighbor_pos[1]]))
                    t = max(t_1, t_2)

                    new_solution = u.swap_x(instance, copy.deepcopy(solution), customer, neighbor, i)
                    t_1_x = u.truck_tour_time(instance.t_t, copy.deepcopy(new_solution[0][0][i]))
                    t_2_x = u.drone_tour_time(instance.t_d, copy.deepcopy(new_solution[0][neighbor_pos[0]][neighbor_pos[1]]))

                    if ((t_1_x < t and t_2_x < t) or (t_1_x < t and t_2_x < t)):
                        return local_search_min_time(instance, new_solution, n_nearest)


            t_old_truck_tour = u.truck_tour_time(instance.t_t, copy.deepcopy(solution[0][0][i]))

            new_solution = u.shift_t(instance, copy.deepcopy(solution), customer, i)
            new_customer_pos = u.get_position(customer, copy.deepcopy(new_solution[0])) #new_customer_pos contiene la pos del customer che è stato spostato dal truck tour al drone tour
            t_new_drone_tour = u.drone_tour_time(instance.t_d, copy.deepcopy(new_solution[0][new_customer_pos[0]][new_customer_pos[1]]))
            
            if (t_new_drone_tour < t_old_truck_tour and new_customer_pos[0] != 0):
                return local_search_min_time(instance, new_solution, n_nearest)

    for i in range(len(solution[0][1])):

        for customer in solution[0][1][i]:
            
            t_old_drone_tour = u.drone_tour_time(instance.t_d, copy.deepcopy(solution[0][1][i]))
            
            new_solution = u.shift_d(instance, copy.deepcopy(solution), customer, i)
            new_customer_pos = u.get_position(customer, copy.deepcopy(new_solution[0])) #new_customer_pos contiene la pos del customer che è stato spostato dal drone tour al truck tour
            t_new_truck_tour = u.truck_tour_time(instance.t_t, copy.deepcopy(new_solution[0][new_customer_pos[0]][new_customer_pos[1]]))
            
            if (t_new_truck_tour < t_old_drone_tour and new_customer_pos[0] != 1):
                return local_search_min_time(instance, new_solution, n_nearest)

    return solution

#Third modification implemented: threshold acceptance considering total completion time
def SISSRs_min_time(instance, sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma, n_nearest, delta, epsilon, iter_imp, iter_max, p_min, p_max, max_unfeasible_swaps_perturb):
    s_0 = h.initial_solution_construction(instance, w1, w2, w3, w4, w5, gamma, n_nearest)
    s_curr = s_0
    s_best = s_0
    iterations_without_improvement = 0
    iteration_counter = 0
    while (iteration_counter < iter_max):
        s = ruin_and_recreate_min_time(instance, copy.deepcopy(s_curr), sigma, c_average_removed, L_max, w1, w2, w3, w4, w5, gamma)
        if u.makespan(instance, s) < u.makespan(instance, s_curr)*(1+delta) or ((u.makespan(instance, s) == u.makespan(instance, s_curr) and u.total_completion_time(instance,s) < u.total_completion_time(instance, s_curr))):
            s_curr = local_search_min_time(instance, s, n_nearest)
            if u.makespan(instance, s_curr) < u.makespan(instance, s_best) or ((u.makespan(instance, s) == u.makespan(instance, s_best) and u.total_completion_time(instance,s) < u.total_completion_time(instance, s_best))):
                s_best = s_curr
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        if iterations_without_improvement >= iter_imp:
            s_curr = h.perturbate(instance, s_curr, p_min, p_max, max_unfeasible_swaps_perturb)
            iterations_without_improvement = 0
        delta = delta * epsilon
        iteration_counter+=1

    return s_best, []