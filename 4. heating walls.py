# https://github.com/Younes-Toumi/
import numpy as np
import matplotlib.pyplot as plt
from time import time
import numba
from numba import jit, cuda

K0 = 273.15

# Defining our problem

a = 0.27
days = 7
duration = 3600*24*days #seconds
nodes = 300

u = np.zeros(nodes) + (20 + K0) # Plate is initially as 25 degres C

# Boundary Conditions 

@numba.jit("f8(f8)", nopython=True, nogil=True, fastmath = True)
def external_temperature(sec):
    h = sec/3600
    h = h%24
    t = 15*np.sin(np.pi*h/12 + 3.9) + 24
    return K0 + t

u[0] = external_temperature(0)
u[-1] = 25 + K0

# Simulating
@numba.jit("(f8[:],f8,f8,f8)", nopython=True, nogil=True, fastmath = True, cache=True)
def solve_heat_eqn(init_state, duration, dt, dx):
    u = init_state.copy()
    counter = 0
    
    inners = []
    # outers = []
    
    while counter < duration :
        w = u.copy()
        for i in range(1, nodes - 1):

            u[i] = dt * a * (w[i - 1] - 2 * w[i] + w[i + 1]) / dx ** 2 + w[i]

        counter += dt
        u[0] = external_temperature(counter)
        u[-1] = u[-2]
        
        inners.append(u[-1])
        # outers.append(u[0])
        
    return u, np.array(inners)



def external_temperature_hour(h):
    h = h%24
    t = 15*np.sin(np.pi*h/12 + 3.9) + 24
    return t

hours_xs = np.linspace(0, 24*days, 300)
plt.plot(hours_xs, external_temperature_hour(hours_xs), 'k', label='Outer wall temperature', alpha=0.3, linewidth=3)


thiccs = [50]#[50, 100, 200, 300, 500]

for thicc in thiccs:
    
    # Initialization 

    length = thicc #mm
    dx = length / (nodes-1)
    dt = 0.5 * dx**2 / a

    start = time()
    print(f'For t = {thicc} mm:\nIterations: {duration/dt}\ndt = {dt}')
    final, inners = solve_heat_eqn(u, duration, dt, dx)
    end = time()
    print('Time elapsed:', end - start, 's\n')
    
    N = len(inners)
    # k = 1
    if N > 1e5:
        k = int(N//1e5)
        inners = inners[::k]
        
    xss = np.linspace(0, duration/3600, len(inners))
    plt.plot(xss, inners-K0, label=f't = {thicc} mm')


plt.xlim(0,24*days)
plt.legend(loc='upper right')
plt.show()
