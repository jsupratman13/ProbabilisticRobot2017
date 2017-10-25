import numpy as np
import copy
import math, random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

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
    def __init__(self, p, w):
        x = 0#random.uniform(-1.0, 1.0)
        y = 0#random.uniform(-0.5, 1.5)
        th = 0#random.uniform(0, math.pi)
        #self.pose = np.array([x, y, th])
        self.pose = np.array(p)
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
    fig = plt.quiver(xs, ys, vxs, vys, color='blue', label="particles")
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
    #sigma = 1.0
    #eval_distance = math.exp(-((distance - rel_distance)**2)/(sigma**2)/2.0)/math.sqrt(2.0*math.pi*(sigma**2))
    #eval_direction = math.exp(-((direction - rel_direction)**2)/(sigma**2)/2.0)/math.sqrt(2.0*math.pi*(sigma**2))

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

## resampling
def resampling(particles):
    sample = []
    particle_num = len(particles)
    pointer = 0.0
    index = int(random.random()*particle_num)
    max_weight = max([e.weight for e in particles])
    for i in range(particle_num):
        pointer += random.uniform(0, 2 * max_weight)
        while particles[index].weight < pointer:
            pointer -= particles[index].weight
            index = (index+1)%particle_num
        particles[index].weight = 1.0/particle_num
        sample.append(copy.deepcopy(particles[index]))
    return sample


def resampling2(particles):
    sample = []
    sm = 0.0
    accum = []
    for p in particles:
        accum.append(p.weight + sm)
        sm += p.weight
    particle_num = len(particles)
    pointer = random.uniform(0.0,1.0/particle_num)
    while pointer < 1.0:
        if accum[0] >= pointer:
            sample.append(Particle(copy.deepcopy(particles[0].pose), 1.0/particle_num))
            pointer += 1.0/particle_num
        else:
            accum.pop(0)
            particles.pop(0)
    return sample

num_particles = 5000
actual_x = np.array([0.0,0.0,0.0]) #actual robot pos
actual_landmarks = [np.array([-0.5, 0.0]), np.array([0.5, 0.0]), np.array([0.0, 0.5])]
particles = [Particle([random.uniform(-1.0,1.0),random.uniform(-0.5,1.5), random.uniform(0, math.pi)],(1.0/num_particles)) for i in range(num_particles)]
#particles = [Particle([0,0,0],(1.0/num_particles)) for i in range(num_particles)]
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
    
    particles = resampling(particles)
    
    particle_path.append(copy.deepcopy(particles))
    
for i, p in enumerate(path):
    draw(path[i], particle_path[i])
    draw_landmarks(actual_landmarks)
    draw_observations(path[i], measurements[i])
    plt.show()
