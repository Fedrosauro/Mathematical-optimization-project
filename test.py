from solver import PDSVRPModel
import time
from instancePDSVRP import instancePDSVRP
import heuristic as h
import heuristic_min_time as hmt
import utilities as u

#test to ensure everything is working 

file_name_instance = "./instances/30-c-0-c.txt" #small instance to test
instance = instancePDSVRP(file_name_instance)

#model test
#print("Model")
#model = PDSVRPModel(instance) 
#model.build_model()
#model.solve()
#model.print_results()

#heuristic test
#print("Heuristic")   
#params =[0.3, instance.N * 0.15, instance.N * 0.15, 5,1,1,2,2, 0.1, 20, 0.1, 0.999975, 100, 1000, (int)(min(3, 0.1*instance.N)), (int)(0.1*instance.N), 9]
#start_time = time.time()
#best_solution = h.SISSRs(instance, *params)
#end_time = time.time()
#elapsed_time_heuristic = end_time - start_time

#print("Best solution found:\n", best_solution)
#print("Solution cost: ", u.cost(instance, best_solution))
#print("Time required for calculation: ", elapsed_time_heuristic)

#heuristic Min Time test
print("Heuristic Min Time")   
params =[0.3, instance.N * 0.15, instance.N * 0.15, 5,1,1,2,2, 0.1, 20, 0.1, 0.999975, 100, 1000, (int)(min(3, 0.1*instance.N)), (int)(0.1*instance.N), 9]
start_time = time.time()
best_solution, _ = hmt.SISSRs_min_time(instance, *params)
end_time = time.time()
elapsed_time_heuristic = end_time - start_time

print("Best solution found:\n", best_solution)
print(f'Solution makespan: {u.makespan(instance, best_solution) * 60:.2f} minutes')
print("Time required for calculation: ", elapsed_time_heuristic)

