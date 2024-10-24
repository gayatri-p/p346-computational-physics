import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

a = 110
length = 10 #mm
time = 1 #seconds
nodes = 40

# Initialization 

dx = length / (nodes-1)
dy = length / (nodes-1)

dt = min(   dx**2 / (4 * a),     dy**2 / (4 * a))

t_nodes = int(time/dt) + 1

u = np.zeros((nodes, nodes)) + 25 # Plate is initially as 20 degres C

# Boundary Conditions 

# u[0, :] = 20 # top
u[:, :] = np.linspace(40, 25, nodes) # bottom
# u[:, 0] = 50 # left
# u[:, -1] = np.linspace(0, 100, nodes) # right

# Visualizing with a plot

fig, axis = plt.subplots()

pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=10, vmax=60)
plt.colorbar(pcm, ax=axis)

# Simulating

counter = 0

def ftemp(time):
    T = 60
    return T

pcm.set_array(u)
axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
plt.show()

def heat_eq2(temp:callable, Lx:float, Nx:int, Lt:float, Nt:int, needed:int):
    """Solves the heat equation in 1D.

    Args:
        temp (callable): Initial temperature distribution.
        Lx (float): Length of the rod.
        Nx (int): Number of grid points in x.
        Lt (float): Time to solve for.
        Nt (int): Number of time steps.
        needed (int): Upto the number of time steps to actually calculate.
    
    Returns:
        A: Approximate solution to the heat equation.
           Where each row is a time step, and each column
           is a point on the rod.
    """
    hx = Lx/Nx
    ht = Lt/Nt
    alpha = ht/(hx**2)
    print(f"{alpha=}")

    A = np.zeros((needed, Nx)).mat
    for i in range(Nx): A[0][i] = temp(Nx, i)
    for t in tqdm(range(1, needed)):
        for x in range(Nx):
            if x == 0:       A[t][x] = 0                 + (1 - 2*alpha)*A[t-1][x] + alpha*A[t-1][x+1]
            elif x == Nx-1:  A[t][x] = alpha*A[t-1][x-1] + (1 - 2*alpha)*A[t-1][x] + 0
            else:            A[t][x] = alpha*A[t-1][x-1] + (1 - 2*alpha)*A[t-1][x] + alpha*A[t-1][x+1]
    
    return A

u = heat_eq2()

pcm.set_array(u)
axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
plt.show()

