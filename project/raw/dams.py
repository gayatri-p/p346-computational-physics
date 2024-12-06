import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

""" Extract flow-rate information from the csv file"""
dates, times, flow = np.loadtxt('project/data/water.csv', unpack = True, usecols = (2, 3, 5), dtype=object)
flow = np.array(flow, dtype=float)
t = np.zeros(dates.size)
for i in range(dates.size):
    d = datetime.strptime(dates[i]+times[i], '%Y-%m-%d%H:%M')
    t[i] = d.timestamp()
start_time = t[0]
t -= start_time

""" Calculate derivatives using central difference approach 
used in the calculation of error"""
def d2dt(y,h=1):
    diffs = np.zeros(len(y))
    for x in range(2, len(y)-2):
        diffs[x] = (y[x+1]+y[x-1]-2*y[x])/(h**2)
    return diffs

def d4dt(y,h=1):
    diffs = np.zeros(len(y))
    for x in range(2, len(y)-2):
        diffs[x] = (y[x+2]-4*y[x+1]+6*y[x]-4*y[x-1]+y[x-2])/h**4
    return diffs

""" Integration functions """
def mid_point_integration(x, y):
    half_step = (x[1]-x[0])/2
    res = 0
    res += y[0]*half_step
    for i in range(1,len(x)-1):
        res += y[i]*half_step*2
    res += y[-1]*half_step

    f2epsilon = np.max(abs(d2dt(y)))
    err = ((x[-1]-x[0]) * f2epsilon * dx**2)/(24)
    return res, err

def trapezoidal_integration(x, y, dx):
    res = (dx/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
    f2epsilon = np.max(abs(d2dt(y)))
    err = ((x[-1]-x[0]) * f2epsilon * dx**2)/(12)
    return res, err

def simpsons_13_integration(x, y, dx):
    # If there are an even number of samples, N, then there are an odd
    # number of intervals (N-1), but Simpson's rule requires an even number
    # of intervals. Hence we perform simpson's rule on the first and last (N-2)
    # intervals, take the average and add up the end points using trapezoidal rule
    if len(x) % 2 == 0:
        res = (dx/3) * (y[0] + 4*np.sum(y[1:-2:2]) + 2*np.sum(y[2:-2:2]) + y[-3])
        res += (dx/3) * (y[1] + 4*np.sum(y[2:-1:2]) + 2*np.sum(y[3:-1:2]) + y[-2])
        res /= 2
        res += 0.5*dx*(y[0] + y[-1])
    else:
        res = (dx/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    f4epsilon = np.max(abs(d4dt(y)))
    err = ((x[-1]-x[0]) * f4epsilon * (dx**4))/(180)
    return res, err

def simpsons_38_integration(x, y, dx):
    # If there are an N number of samples, then there are an (N-1)
    # number of intervals. Simpson's 3/8 rule requires an 3n number
    # of intervals. Hence in case of 3n-1 or 3n-2 intervals, we approximate the end points
    # similar to what we did for Simpson's 1/3 rule
    if len(x) % 3 == 0: # (n-1)%3 = 2
        res = y[0] + 3*(np.sum(y[1:-3:3])+np.sum(y[2:-3:3])) + 2*np.sum(y[3:-3:3])+ y[-3]
        res += y[1] + 3*(np.sum(y[2:-2:3])+np.sum(y[3:-2:3])) + 2*np.sum(y[4:-2:3])+ y[-2]
        res += y[2] + 3*(np.sum(y[3:-1:3])+np.sum(y[4:-1:3])) + 2*np.sum(y[5:-1:3])+ y[-1]
        res *= (3*dx/8) * (1/3)
        res += dx*(y[0] + y[-1])
    elif len(x) % 3 == 1: # (n-1)%3 = 0
        res = y[0] + 3*(np.sum(y[1:-1:3])+np.sum(y[2:-1:3])) + 2*np.sum(y[3:-1:3])+ y[-1]
        res *= (3*dx/8)
    elif len(x) % 3 == 2: #(n-1)%3 = 1
        res = y[0] + 3*(np.sum(y[1:-2:3])+np.sum(y[2:-2:3])) + 2*np.sum(y[3:-2:3])+ y[-2]
        res += y[1] + 3*(np.sum(y[2:-1:3])+np.sum(y[3:-1:3])) + 2*np.sum(y[4:-1:3])+ y[-1]
        res *= (3*dx/8) * (1/2)
        res += 0.5*dx*(y[0] + y[-1])

    f4epsilon = np.max(abs(d4dt(y)))
    err = ((x[-1]-x[0]) * f4epsilon * (dx**4))/(80)
    return res, err

""" Convert the domain to [0, 1] for simplicity in calculation of errors 
and then multiply by t_final (t_inital = 0)"""
tmax = t[-1]-t[0]
int_t = t/np.max(t)
dx = int_t[1]-int_t[0]

mid_point, max_error_mid_point = mid_point_integration(int_t, flow)
trapezoidal, max_error_trapz = trapezoidal_integration(int_t, flow, dx)
simpsons_13, max_error_simpsons13 = simpsons_13_integration(int_t, flow, dx)
simpsons_38, max_error_simpsons38 = simpsons_38_integration(int_t, flow, dx)

print(f'Mid-point: {mid_point*tmax*1e-9:.3f} x 10^9 \pm {max_error_mid_point*tmax*(900**2)*1e-5:.3f} x 10^5 ft^3')
print(f'Trapezoidal: {trapezoidal*tmax*1e-9:.3f} x 10^9 \pm {max_error_trapz*tmax*(900**2)*1e-5:.3f} x 10^5 ft^3')
print(f'Simpsons 1/3: {simpsons_13*tmax*1e-9:.3f} x 10^9 \pm {max_error_simpsons13*tmax*(900**4):.3f} ft^3')
print(f'Simpsons 3/8: {simpsons_38*tmax*1e-9:.3f} x 10^9 \pm {max_error_simpsons38*tmax*(900**4):.3f} ft^3')