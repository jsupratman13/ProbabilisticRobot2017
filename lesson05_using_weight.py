import numpy as np
import copy
import math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

actual_landmarks = [np.array([-0.5, 0.0]), np.array([0.5, 0.0]), np.array([0.0, 0.5])]
actual_x = np.array([0.3, 0.2, math.pi*20.0/180])

def draw_landmarks(landmarks):
    xs = [e [0] for e in landmarks]
    ys = [e [1] for e in landmarks]
    plt.scatter(xs, ys, s=300, marker="*", label="landmarks", color="orange")

def draw_observation(pose, measurement):
    x,y,th = pose
    distance, direction, lx, ly = measurement
    lx = distance * math.cos(th + direction) + x
    ly = distance * math.sin(th + direction) + y
    plt.plot([x, lx], [y, ly], color='pink')
    
def draw_observations(pose, measurement):
    for m in measurement:
        draw_observation(pose, m)    

def relative_landmark_pos(pose, landmark):
    x,y,th = pose
    lx, ly = landmark
    distance = math.sqrt((x-lx)**2 + (y-ly)**2)
    direction = math.atan2(ly-y, lx-x) - th

    return (distance, direction, lx, ly)

## adding noise
def observation(pose, landmark):
    actual_distance, actual_direction, lx, ly = relative_landmark_pos(pose, landmark)
    #place limit in sensor view
    if math.cos(actual_direction) < 0.0: #only see left right 90deg, cant see anything behind
        return None

    measured_distance = random.gauss(actual_distance, actual_distance*0.1) #10% error
    measured_direction = random.gauss(actual_direction, 5.0/180.0*math.pi) #5deg error
    return (measured_distance, measured_direction, lx, ly)

# remove None from list
def observations(pose, landmarks):
    return filter(lambda x: x != None, [observation(pose,e) for e in landmarks])

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
    col = [e.weight for e in particles]
    fig = plt.quiver(xs, ys, vxs, vys, col, label="particles")
    plt.quiver([pose[0]], [pose[1]], [math.cos(pose[2])], [math.sin(pose[2])],color="red",label="actual robot motion")

## comparing particle and measurements
def likelihood(pose, measurement):
    x,y,th = pose
    distance, direction, lx, ly = measurement

    # prediction of measurement from particles
    rel_distance, rel_direction, tmp_x, tmp_y = relative_landmark_pos(pose, (lx,ly))
    
    # evaluate error using gauss
    eval_distance = norm.pdf(x = distance - rel_distance, loc = 0.0, scale = rel_distance/10.0)
    eval_direction = norm.pdf(x = direction - rel_direction, loc = 0.0, scale = 5.0/180.0 * math.pi)
    return eval_distance * eval_direction

## update particle weights
def change_weights(particles, measurement):
    for p in particles:
        p.weight *= likelihood(p.pose, measurement)

    # normalize weight (making sure the total weight equals to 1
    ws = [p.weight for p in particles]
    s = sum(ws)
    for p in particles:
        p.weight = p.weight/s

actual_x = np.array([0.0,0.0,0.0]) #actual robot pos
particles = [Particle(1.0/100) for i in range(100)] #100 particles
u = np.array([0.2, math.pi/180.0 * 20]) #robot motion

path = [actual_x]
particle_path = [copy.deepcopy(particles)]
measurements = [observations(actual_x, actual_landmarks)]

for i in range(10):
    actual_x = f(actual_x, u)
    path.append(actual_x)
    ms = observations(actual_x, actual_landmarks)
    measurements.append(ms)
    
    for p in particles:
        p.pose = f(p.pose, u)

    for m in ms:
        change_weights(particles, m)

    particle_path.append(copy.deepcopy(particles))
    
for i, p in enumerate(path):
    draw(path[i], particle_path[i])
    draw_landmarks(actual_landmarks)
    draw_observations(path[i], measurements[i])
    plt.show()
