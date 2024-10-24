import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

a = 110*60 # mm^2 / min
length = 10 # mm
time = 5 # mins
nodes = 40

# Initialization 

dx = length / (nodes-1)
dy = length / (nodes-1)

dt = min(   dx**2 / (4 * a),     dy**2 / (4 * a))
print(dt)
t_nodes = int(time/dt) + 1

u = np.zeros((nodes, nodes)) + 25 # Plate is initially as 20 degres C

# Boundary Conditions 

# u[0, :] = 20 # top
# u[:, :] = np.linspace(100, 25, nodes) # bottom
# u[:, 0] = 50 # left
# u[:, -1] = np.linspace(0, 100, nodes) # right

# Visualizing with a plot

# fig, axis = plt.subplots()
# pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=10, vmax=60)
# plt.colorbar(pcm, ax=axis)

# Simulating

counter = 0

def ftemp(time):
    T = 100
    return T

# pcm.set_array(u)
# axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
# plt.pause(1e-10)
# plt.show()

total_time_steps = int(time/dt)+1
pbar = tqdm(total = total_time_steps)
snap_interval = total_time_steps//9
snap_us = []
snap_counter = 0

while counter < time :

    w = u.copy()

    for i in range(1, nodes - 1):
        for j in range(1, node  s - 1):

            dd_ux = (w[i-1, j] - 2*w[i, j] + w[i+1, j])/dx**2
            dd_uy = (w[i, j-1] - 2*w[i, j] + w[i, j+1])/dy**2

            u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]

    counter += dt
    snap_counter += 1
    u[:0] = ftemp(counter)
    pbar.update(1)

    if snap_counter == snap_interval:
        snap_counter = 0
        snap_us.append(u)
    # print("t: {:.3f} [s], Avg. temperature of inner wall: {:.2f} Celcius".format(counter, np.average(u[:, -1])))

    # Updating the plot

    # pcm.set_array(u)
    # axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
    # plt.pause(1e-10)
    # break

pbar.close()
print(len(snap_us))

fig, ax = plt.subplots(nrows=3, ncols=3, sharey=False, sharex=False)
fig.set_figheight(18)
fig.set_figwidth(15)

for i in range(3):
    for j in range(3):
        # ax[i][j].set_title("Distribution at t: {:.3f} [s].".format(counter))
        # ax[i][j].plot(snap_us[i+j])
        pcm = ax[i][j].pcolormesh(u, cmap=plt.cm.jet, vmin=10, vmax=60)
        pcm.set_array(snap_us[i+j])
        plt.colorbar(pcm, ax=ax[i][j])

plt.show()

