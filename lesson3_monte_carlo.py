import numpy as np
import copy, math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Particle(object):
    def __init__(self, w):
        self.pose = np.array([0.0, 0.0, 0.0])
        self.weight = w

    def __repr__(self):
        return "pose: " + str(self.pose) + " weight: " + str(self.weight)

def f(x_old, u):
    pos_x, pos_y, pos_th = x_old
    act_fw, act_rot = u
    
    act_fw = random.gauss(act_fw, act_fw/10) #adjust noise 20, 2
    dir_error = random.gauss(0, math.pi/180*3) #adjust noise 
    act_rot = random.gauss(act_rot, act_rot/10) #adjust noise 20, 2
    
    pos_x += act_fw * math.cos(pos_th + dir_error)
    pos_y += act_fw * math.sin(pos_th + dir_error)
    pos_th += act_rot

    return np.array([pos_x, pos_y, pos_th])

def draw(pose, particles):
    fig = plt.figure(i, figsize=(8,8))
    sp = fig.add_subplot(111, aspect='equal')
    sp.set_xlim(-1.0, 1.0)
    sp.set_ylim(-0.5, 1.5)

    xs = [e.pose[0] for e in particles]
    ys = [e.pose[1] for e in particles]
    vxs = [math.cos(e.pose[2]) for e in particles]
    vys = [math.sin(e.pose[2]) for e in particles]
    plt.quiver(xs, ys, vxs, vys, color="blue", label="particles")
    plt.quiver([pose[0]], [pose[1]], [math.cos(pose[2])], [math.sin(pose[2])],color="red",label="actual robot motion")
    plt.show()

actual_x = np.array([0.0, 0.0, 0.0])
u = np.array([0.2, math.pi/180.0*20.0])
particles = [Particle(0.01) for i in range(100)]
print particles[0]

path = [actual_x]
particle_path = [copy.deepcopy(particles)]
for i in range(10):
    actual_x = f(actual_x, u)
    path.append(actual_x)

    for p in particles:
        p.pose = f(p.pose, u)
    particle_path.append(copy.deepcopy(particles))

#print path[0]
#print particle_path[0]
#print path[10]
#print particle_path[10]

for i,p in enumerate(path):
    draw(path[i], particle_path[i])

