# problem 5b

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib import cm
import numba
from numba import jit

# Initialise space grid with 50 random spawns
n = 100
i = 5 # to avoid spawing any population near the edge 
coords = np.array([(np.random.randint(i, n-i), np.random.randint(i, n-i)) for _ in range (50)])
init_population = np.zeros((n, n))
for x, y in coords:
    init_population[x,y] = 2 # arbitrary value

# diffusion constant
D = 0.01

# set the dimensions of the problem
x = 1
dx = 0.05
dt = 0.0001 # such that D*dt/dx**2 < 1/4

times = 252000 # number of iterations
times_snapshot = 3600 # total number of snapshots
f = int(times/times_snapshot)

population_frames = np.zeros([times_snapshot, 100, 100])
population_frames[0] = init_population
population_density = np.zeros(times) # keeps track of the average population density

# Solving the PDE
# Set up numba function
@numba.jit("(f8[:,:,:], f8, f8, f8)", nopython=True, nogil=True, fastmath = True)
def solve_pde(environment, K, r, h):
    cs = environment[0].copy() #current state
    length = len(cs[0])
    density = np.zeros(times)
    density[0] = np.average(cs) # average population density
    cf = 0 # current frame

    for t in range(1, times):
        ns = cs.copy() # new state

        # Since only iterate spatially from 1 to n-1
        # the algorithm by design is implementing Dirichlet BCs
        for i in range(1, length-1):
            for j in range(1, length-1):
                growth = dt*((r-h)*cs[j][i] - (r*cs[j][i]**2)/K)
                diffusion = D*dt/dx**2 * (cs[j+1][i] + cs[j-1][i] +\
                                                    cs[j][i+1] + cs[j][i-1] -\
                                                    4*cs[j][i])
                ns[j][i] = cs[j][i] + diffusion + growth

        # Implementing Neumann BCs
        ns[:,0] = ns[:,1] # left boundary
        ns[:,-1] = ns[:,-2] # right boundary
        ns[0,:] = ns[1,:] # top boundary
        ns[-1,:] = ns[-2,:] # bottom boundary
        
        density[t] = np.average(cs)
        cs = ns.copy()
        if t%f==0: # take snapshot
            cf = cf + 1
            environment[cf] = cs
            
    return environment, density

# Setting up the parameters
K, r, h = 1, 0.9, 0.2

# Get population snapshots and population size over time and plot
population_frames, population_sizes = solve_pde(population_frames, K, r, h)
plt.plot(np.linspace(0, times*dt, times), population_sizes)
plt.xlabel('Time (s)')
plt.ylabel('Total Population')
plt.show()

# generate an animation of the simulation over time
def animate(i):
    ax.clear()
    im = ax.contourf(population_frames[10*i], 100, levels=np.linspace(0,np.max(population_sizes),50))
    plt.title(f't = {10*i*f*dt:.2f} sec')
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    return fig,

fig, ax = plt.subplots(figsize=(8,6))
fig.colorbar(im, ax=ax)
ani = animation.FuncAnimation(fig, animate,
                               frames=359, interval=50)
ani.save('simulation.gif', writer='pillow', fps=30)

# 5c

def gauss2d(x, y, cx=0.5, cy=0.5):
    # 2D gaussian in domain [0,1]x[0,1]
    # with peak at (cx, cy)
    z = np.exp(-(x-cx)**2-(y-cy)**2)
    return z

def gauss2d_inv(x, y, cx=0.5, cy=0.5):
    # Inverted gaussian function
    z = 1-np.exp(-(x-cx)**2-(y-cy)**2)
    return z

def long_hill_fn(x,y):
    # fn with an extended horizontal peak 
    return np.sin((x+3)*(y-0.5)**2)

terrain = np.zeros((n,n))
X    = np.linspace(0, 1, n)
Y    = np.linspace(0, 1, n)
X, Y = np.meshgrid(X, Y)

terrain1 = gauss2d(X,Y)
terrain2 = gauss2d_inv(X,Y)
terrain3 = gauss2d_inv(X,Y, cx=0.2, cy=0.6)
terrain4 = long_hill_fn(X,Y)

# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,6), sharey=True, sharex=True)

# terrains = [terrain1, terrain2, terrain3, terrain4]
# for ax, terrain in zip(axes.flat, terrains):
#     terrain_contour = ax.contour(X,Y,terrain,colors='black')
#     ax.clabel(terrain_contour, inline=True, fontsize=8)
#     color = ax.imshow(terrain, extent=[0, 1, 0, 1], origin='lower', cmap='turbo', alpha=0.8, vmax=1, vmin=0)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

# cax = fig.add_axes([0.95, 0.1, 0.04, 0.75])
# cbar = fig.colorbar(color, cax=cax, orientation='vertical')
# cbar.ax.set_ylabel('Terrain Elevation')
# cbar.ax.set_yticks(ticks=np.linspace(1,0,3))
# cbar.ax.set_yticklabels(np.linspace(1,0,3), rotation='vertical', va='center')
# plt.show()

@numba.jit("(f8[:,:,:], f8, f8, f8)", nopython=True, nogil=True, fastmath = True)
def solve_pde(environment, K, r, h):
    cs = environment[0].copy() #current state
    length = len(cs[0])
    density = np.zeros(times)
    density[0] = np.average(cs) # average population density
    cf = 0 # current frame

    for t in range(1, times):
        ns = cs.copy() # new state

        # Since only iterate spatially from 1 to n-1
        # the algorithm by design is implementing Dirichlet BCs
        for i in range(1, length-1):
            for j in range(1, length-1):
                growth = dt*((r-h)*cs[j][i] - (r*cs[j][i]**2)/K)
                diffusion = (1/2*dx**2)* (D[j,i] *(cs[j, i-1]+cs[j, i+1]+cs[j-1,i]\
                                                   +cs[j+1,i]-4*cs[j,i]) +\
                                         D[j-1,i]*(cs[j-1,i]-cs[j,i])+\
                                         D[j+1,i]*(cs[j+1,i]-cs[j,i])+\
                                         D[j,i-1]*(cs[j,i-1]-cs[j,i])+\
                                         D[j,i+1]*(cs[j,i+1]-cs[j,i]))
                ns[j][i] = cs[j][i] + growth + diffusion

        # Implementing Neumann BCs
        ns[:,0] = ns[:,1] # left boundary
        ns[:,-1] = ns[:,-2] # right boundary
        ns[0,:] = ns[1,:] # top boundary
        ns[-1,:] = ns[-2,:] # bottom boundary
        
        density[t] = np.average(cs)
        cs = ns.copy()
        if t%f==0: # take snapshot
            cf = cf + 1
            environment[cf] = cs
            
    return environment, density