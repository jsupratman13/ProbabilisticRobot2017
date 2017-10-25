import numpy as np
import math, random
import matplotlib.pyplot as plt

def gaussian(mu, sigma, x):
    return 1/math.sqrt(2.0*math.pi*sigma)*math.exp(-0.5*(x-mu)**2/sigma)

def f(old_x, u):
    x, y, th = old_x
    distance, orientation = u

    error = 0.1

    distance = random.gauss(distance, distance*error)
    orientation = random.gauss(orientation, orientation*error)
    dir_error = random.gauss(0.0, math.pi/180.0*3.0)

    x = x + distance * math.cos(th + dir_error)
    y = y + distance * math.sin(th + dir_error)
    th = th + orientation
    
    return np.array([x, y, th])

x = np.array([0.0, 0.0, 0.0])
u = np.array([0.1, 10.0/180*math.pi])

#transition example
print(x)
path = []
for i in range(1000):
    x = f(x,u)
    print x
    path.append(x)

#draw figure
fig = plt.figure(i, figsize=(8,8))
sp = fig.add_subplot(111, aspect='equal')
sp.set_xlim(-1.0,1.0)
sp.set_ylim(-0.5, 1.5)
xs = [e[0] for e in path]
ys = [e[1] for e in path]
vxs = [math.cos(e[2]) for e in path]
vys = [math.sin(e[2]) for e in path]
plt.quiver(xs,ys,vxs,vys,color='red',label="path")
plt.show()
