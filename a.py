import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.integrate as sp

dates, times, flow = np.loadtxt('water.csv', unpack = True, usecols = (2, 3, 5), dtype=object)
flow = np.array(flow, dtype=float)
t = np.zeros(dates.size)
for i in range(dates.size):
    d = datetime.strptime(dates[i]+times[i], '%Y-%m-%d%H:%M')
    t[i] = d.timestamp()
start_time = t[0]
t -= start_time

fig, ax = plt.subplots(figsize = (5,5))
# ax2 = ax.twiny()
# ax2.plot(x, y2, color = 'b')
ax.plot(t, flow, 'k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow Rate (ft$^3$ s$^{-1}$)')
ax.set_ylim(0, 6400)
ax.set_xlim(0, t[-1])
plt.tight_layout()
plt.show()