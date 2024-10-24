import numpy as np
import matplotlib.pyplot as plt


# Defining our problem

a = 110
length = 30 #mm
time = 8 #seconds
nodes = 20

# Initialization 

dx = length / (nodes-1)
dt = 0.5 * dx**2 / a
t_nodes = int(time/dt) + 1

u = np.zeros(nodes) + 25 # Plate is initially as 20 degres C

# Boundary Conditions 

def f(t):
    return 100*np.sin(2*np.pi*t/4)+20

u[0] = f(0)
u[-1] = 25


# Visualizing with a plot

fig, axis = plt.subplots()

pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=-80, vmax=120)
plt.colorbar(pcm, ax=axis)
axis.set_ylim([0, 1])

# Simulating

counter = 0
i = 0 # 3
os = []
iss = []
while counter < time :

    w = u.copy()

    for i in range(1, nodes - 1):

        u[i] = dt * a * (w[i - 1] - 2 * w[i] + w[i + 1]) / dx ** 2 + w[i]

    counter += dt
    u[0] = f(counter)
    os.append(u[0])
    iss.append(u[-2])
    print("t: {:.3f} [s], Inside temperature: {:.2f} Celcius".format(counter, u[-1]))

    # Updating the plot

    pcm.set_array([u])
    axis.set_title("Distribution at t: {:.3f} [s]\nInside temperature: {:.2f} Celcius".format(counter, u[-2]))
    axis.set_ylabel('x')
    plt.pause(0.1)
    i += 1

plt.show()
print(i)
ts = np.linspace(0,4,len(os))
plt.plot(ts,os,label='T_{outer}')
plt.plot(ts,iss,label='T_{inner}')
plt.xlabel('Time(s)')
plt.ylabel('Temerature (Celcius)')
plt.show()








