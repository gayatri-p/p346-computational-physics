import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.integrate as sp

dates, times, flow = np.loadtxt('data/water.csv', unpack = True, usecols = (2, 3, 5), dtype=object)
flow = np.array(flow, dtype=float)
t = np.zeros(dates.size)
for i in range(dates.size):
    d = datetime.strptime(dates[i]+times[i], '%Y-%m-%d%H:%M')
    t[i] = d.timestamp()
start_time = t[0]
t -= start_time

def f(val, xs=t, ys=flow):
    if len(val) == 1:
        idx = (np.abs(xs - val)).argmin()
        return ys[idx]
    else:
        ids = np.zeros(len(val), dtype=int)
        for i in range(len(val)):
            ids[i] = np.argmin(np.abs(xs - val[i])).astype(int)
        return ys[ids]
    
fig, ax = plt.subplots(figsize = (10,5))

ax.plot(t, flow, 'k')
tdash = np.linspace(t[0], t[-1], 1000)
ax.plot(tdash, f(tdash), 'r')
# ax.plot(t[1:], np.diff(flow), 'k')
# ax.plot(t[1:], np.diff(t), 'xk')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow Rate (ft$^3$ s$^{-1}$)')
# ax.set_ylim(0, 6400)
ax.set_xlim(0, t[-1])

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(np.linspace(0, t[-1], 12))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax2.set_xlabel('Month of the Year')
plt.tight_layout()
plt.show()