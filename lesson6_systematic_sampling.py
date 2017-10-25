import numpy as np
import math, random, copy
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, p, w):
        self.pose = np.array(p)
        self.weight = w

    def __repr__(self):
        return 'pose: ' + str(self.pose) +' weight: ' + str(self.weight)

particles = [Particle([1.0, 0.0, 0.0], 0.1),
             Particle([2.0, 0.0, 0.0], 0.2),
             Particle([3.0, 0.0, 0.0], 0.3),
             Particle([4.0, 0.0, 0.0], 0.4)]

for p in particles:
    print p

accum = []
sm = 0.0
for p in particles:
    accum.append(p.weight + sm)
    sm += p.weight

print accum

pointer = random.uniform(0.0, 1.0/len(particles))
print pointer

new_particles = []
particles_num = len(particles)

while pointer < 1.0:
    if accum[0] >= pointer:
        new_particles.append(Particle(copy.deepcopy(particles[0].pose),1.0/particles_num))
        pointer += 1.0/particles_num
    else:
        accum.pop(0)
        particles.pop(0)

for p in new_particles:
    print p
particles = new_particles
