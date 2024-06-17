import gurobipy as gb

class PDSVRPModel:

    def __init__(self, instance):
        self.C_T = instance.C_T #truck transportation cost
        self.C_D = instance.C_D #drone transportation cost
        self.D = instance.D #number of drones
        self.N = instance.N #number of nodes
        self.h = instance.h #number of trucks
        self.distances= instance.distances #matrix of euclidean distances
        self.manhattan_distances = instance.manhattan_distances #matrix of manhattan distances
        self.t_t = instance.t_t #matrix of trucks travel times
        self.t_d = instance.t_d #vector of drones travel times
        self.Q_t = instance.Q_t #truck capacity
        self.Q_d = instance.Q_d #drone capacity
        self.T_t = instance.T_t #max time truck
        self.T_d = instance.T_d #max time drones
        self.w = instance.w #weights vector
        self.d_end = instance.d_end #drone endurance
        
        self.model = gb.Model("OptimizationModel")
        #decision variables
        self.x = None
        self.y = None
        self.z = None
        self.u = None

        
    def build_model(self):
        # Define the decision variables
        self.x = self.model.addVars([(i,j) for i in range(self.N) for j in range(self.N) if i!=j], vtype=gb.GRB.BINARY, name="x")
        self.y = self.model.addVars([(i,k) for i in range(1, self.N) for k in range(self.D)], vtype=gb.GRB.BINARY, name="y")
        self.z = self.model.addVars([(i,j) for i in range(self.N) for j in range(self.N) if i!=j], lb=0, ub=self.T_t, vtype=gb.GRB.CONTINUOUS, name="z") 
        self.u = self.model.addVars([(i) for i in range(1, self.N)], lb=0, ub=self.Q_t, vtype=gb.GRB.CONTINUOUS, name="u")

        # Define the objective function
        self.model.setObjective(
            gb.quicksum(self.C_T * self.manhattan_distances[i,j] * self.x[i, j] for i in range (self.N) for j in range (self.N) if i != j) +
            gb.quicksum(self.C_D * 2 * self.distances[0,k] * self.y[k, l] for k in range (1, self.N) for l in range(self.D)),
            gb.GRB.MINIMIZE
        ) #ok

        N_t = [i for i in range(1,self.N) if (self.w[i] > self.Q_d or ((self.t_d[i] * 2) > self.d_end))] #indexes of clients that must be served by trucks
        N_f= [i for i in range(1,self.N) if i not in N_t] #indexes of clients that can ber served either by e truck or a drone

        # Add constraints
        self.model.addConstr(gb.quicksum(self.x[0, i] for i in range (1, self.N)) <= self.h, "2) Constraint on number of trucks") 

        for j in range(self.N):
            self.model.addConstr(gb.quicksum(self.x[i, j] for i in range(self.N) if i != j) == gb.quicksum(self.x[j, i] for i in range(self.N) if i != j), "3) Flow constraint") 

        for j in N_f:
            self.model.addConstr(gb.quicksum(self.x[i, j] for i in range(self.N) if i != j) + gb.quicksum(self.y[j, k] for k in range(self.D)) == 1, "4) Every customer in N_f must be served") 

        for j in N_t:
            self.model.addConstr(gb.quicksum(self.x[i, j] for i in range(self.N) if i != j) == 1, "5) Truck customers must be visited by only trucks")

        for i in range (1, self.N):
            for j in range (1, self.N):
                if j != i:
                    self.model.addConstr(self.u[i] - self.u[j] + self.Q_t * self.x[i, j] <= self.Q_t - self.w[j], "6) Miller-Tucker-Zemlin constraint")

        for k in range(self.D):
            self.model.addConstr(gb.quicksum(self.y[j, k] * self.t_d[j] for j in N_f) <= self.T_d, "7) Time constraint for drones")

        for i in range(1, self.N):
            self.model.addConstr(gb.quicksum(self.z[l, i] for l in range(self.N) if l != i) + gb.quicksum(self.t_t[i, j] * self.x[i, j] for j in range(self.N) if j != i) == gb.quicksum(self.z[i, j] for j in range(self.N) if j != i),"8) Inductive step for induction method of cumulative time additions")

        for i in range(1, self.N):
            self.model.addConstr(self.z[0, i] == self.t_t[0, i] * self.x[0, i], "9) Basic step for induction method of cumulative time additions")

        for i in range(1, self.N):
            self.model.addConstr(self.z[i, 0] <= self.T_t * self.x[i, 0], "10) Truck time limit constraint")

        # Optimization parameters
        self.model.setParam('Threads', 8) #Set number of threads
        self.model.setParam('MIPFocus', 1)  # Set focus
        self.model.setParam('Presolve', 2)  # Presolve level increase
        self.model.setParam('Cuts', 2)  # Aggressive cuts
        self.model.setParam('TimeLimit', 3600) # Time limit of an hour

    def solve(self):
        self.model.optimize()

    def print_results(self):
        if self.model.status == gb.GRB.OPTIMAL:
            for v in self.model.getVars():
                if v.x != 0:
                    print(f'{v.varName}: {v.x}')
            print("Non printed variables equal 0")
            print(f'Obj: {self.model.objVal}')
            print(f'Time taken: {self.model.Runtime} seconds')
        else:
            print("No optimal solution found.")
            print(f'Time taken: {self.model.Runtime} seconds')