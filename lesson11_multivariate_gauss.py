import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse

### CREATE GAUSS DISTRIBUTION
'''
# define mean and covariance matrix
mean = np.array([0.0, 1.0])
cov = np.array([[1.0,0.5],  #adjust covariance to see changes 0 <= cov < 1
                [0.5,1.0]])

# create mesh grid
x_axis = np.linspace(-5,5,100)
y_axis = np.linspace(-5,5,100)
X,Y = np.meshgrid(x_axis, y_axis)

#bivariate normal
Z = mlab.bivariate_normal(X,Y, # lattice coordinate
                          math.sqrt(cov[0][0]),math.sqrt(cov[1][1]), # standard deviation
                          mean[0],mean[1], # mean
                          cov[0][1]) # covariance

plt.contour(X,Y,Z)
plt.axis('equal')
plt.show()
'''

### CREATING CLASS
class Gaussian2D(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov  =cov

    def value(self, x, y):
        pos = np.array([[x],[y]])
        delta = pos - self.mean
        numerator = math.exp(-0.5 * (delta.T).dot(np.linalg.inv(self.cov)).dot(delta))
        denominator  = 2 * math.pi * math.sqrt(np.linalg.det(self.cov))
        return numerator / denominator

    def shift(self, delta, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.array([[c,-s],
                        [s, c]])

        self.cov = rot.dot(self.cov).dot(rot.T) # cov = rot * cov * rot.T
        self.mean = self.mean + delta
    
    # error ellipse (use eigen vector, eigen value as major,minor axis in ellipse)
    def ellipse(self, color="blue"):
        eig_val, eig_vec = np.linalg.eig(self.cov)
        v1 = eig_val[0] * eig_vec[:,0]
        v2 = eig_val[1] * eig_vec[:,1]
        v1_direction = math.atan2(v1[1],v1[0])

        e = Ellipse(self.mean,width=np.linalg.norm(v1),height=np.linalg.norm(v2),angle=v1_direction/math.pi*180)
        e.set_facecolor(color)
        e.set_alpha(0.2)
        return e

'''
mean = np.array([[1.0],[3.0]])
cov = np.array([[1.0,0.5],
                [0.5,1.0]])
p = Gaussian2D(mean, cov)
fig = plt.figure(0)
sp = fig.add_subplot(111,aspect='equal')
plt.xlim(0,5)
plt.ylim(0,5)
sp.add_artist(p.ellipse())

p.shift(np.array([[1.0],[0.0]]),-math.pi/4.0)
fig = plt.figure(1)
sp = fig.add_subplot(111,aspect='equal')
plt.xlim(0,5)
plt.ylim(0,5)
sp.add_artist(p.ellipse())
plt.show()
'''

def multi(A,B):
    invA = np.linalg.inv(A.cov)
    invB = np.linalg.inv(B.cov)
    cov = np.linalg.inv(invA+invB) 
    
    K = cov.dot(invB)
    mean = (np.identity(2)-K).dot(A.mean)+K.dot(B.mean)
    return Gaussian2D(mean,cov)

p = Gaussian2D(np.array([[1.0],[3.0]]),np.array([[1.0,0.5],[0.5,1.0]]))
#q = Gaussian2D(np.array([[1.0],[3.0]]),np.array([[1.0,0.5],[0.5,1.0]]))
q = Gaussian2D(np.array([[4.0],[1.0]]),np.array([[1.0,-0.5],[-0.5,1.0]]))

r = multi(p,q)

fig = plt.figure(0)
sp = fig.add_subplot(111,aspect='equal')
plt.xlim(0,5)
plt.ylim(0,5)
sp.add_artist(p.ellipse('blue'))
sp.add_artist(q.ellipse('yellow'))
sp.add_artist(r.ellipse('red'))
plt.show()

## Easy method using scipy
from scipy.stats import multivariate_normal
rv = multivariate_normal([0.5, -0.2],[[2.0,0.3],[0.3,0.5]])
x,y = np.mgrid[-1:1:0.01, -1:1:0.01]
pos = np.empty(x.shape+(2,))
pos[:,:,0] = x
pos[:,:,1] = y
plt.contourf(x,y,rv.pdf(pos))
plt.show()
