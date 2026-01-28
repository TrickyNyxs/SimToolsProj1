import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apro
import matplotlib.pyplot as mpl
import assimulo.solvers as asol
import BDF4Solver
import EESolver
import BDF2Solver

# Paramters for the elastic pendulum
m = 1.0      # mass
k = 10    # spring constant
L0 = 1.0     # natural length of the spring
g = 1.0     # acceleration due to gravity


# Define Problem 
initial_conditions = [0.0, 1.1, 0.1, 0.0]  # initial position (x,y) and velocity (vx, vy)
k_list = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
EE_error_list = []
BDF4_error_list = []
BDF2_error_list = []
simulation_time = 4.9

for k in k_list:
    print(k)
    def elastic_pendulum(t,y):
        yvec = np.zeros_like(y)
        yvec[0] = y[2]
        yvec[1] = y[3]
        lam = k * (np.sqrt(y[0]**2 + y[1]**2) - 1) / np.sqrt(y[0]**2 + y[1]**2)
        yvec[2] = -1*y[0]*lam
        yvec[3] = -1*y[1]*lam - 1
        return yvec
    
    eP_Problem = apro.Explicit_Problem(elastic_pendulum, t0 = 0, y0 = initial_conditions)
    
    eP_Solver = asol.CVode(eP_Problem)
    eP_Solver.reset() # Why is this needed here?
    t_sol, y_sol = eP_Solver.simulate(simulation_time, 491) # simulate(tf, ncp)

    # EE Solver
    exp_sim = EESolver.Explicit_Euler(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    t, y = exp_sim.simulate(simulation_time, 1000)
    error = np.linalg.norm(y_sol - y, ord=np.inf)
    EE_error_list.append(error)

    # BDF4 Solver
    exp_sim = BDF4Solver.BDF4(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    t, y = exp_sim.simulate(simulation_time, 1000)
    error = np.linalg.norm(y_sol - y, ord=np.inf)
    BDF4_error_list.append(error)

    # BDF2 Solver
    exp_sim = BDF2Solver.BDF_2(eP_Problem) #Create a BDF solver
    exp_sim.reset()
    t, y = exp_sim.simulate(simulation_time, 1000)
    error = np.linalg.norm(y_sol - y, ord=np.inf)
    BDF2_error_list.append(error)
    


plt.figure()
plt.loglog(k_list, EE_error_list, marker='o')
plt.loglog(k_list, BDF4_error_list, marker='x')
plt.loglog(k_list, BDF2_error_list, marker='s')

plt.xlabel('Spring Constant k')
plt.ylabel('Norm of Error')
plt.title('Error vs Spring Constant for Elastic Pendulum')
plt.grid(True, which="both", ls="--")
plt.show()

    
   