import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

size = 3 #size of map
values = np.array([[100.0,100.0,100.0],[100.0,100.0,100.0],[100.0,100.0,0.0]]) #initial value for each grid
swamp = (2,1)
goal = (2,2) #set goal state
actions = ['up','down','left','right'] #actions robot can take
policy = [['up','up','up'],['up','up','up'],['up','up',None]] #brain of robot

def draw(mark_pos, action):
    fig, ax = plt.subplots()
    mp = ax.pcolor(values, cmap=plt.cm.YlOrRd, vmin=0, vmax=8)
    ax.set_aspect(1)
    ax.set_xticks(range(size), minor=False)
    ax.set_yticks(range(size), minor=False)

    for x in range(len(values)):
        for y in range(len(values[0])):
            plt.text(x+0.5,y+0.5,round(values[x][y],2),ha='center',va='center',size=20)

    plt.text(goal[0]+0.75,goal[1]+0.75,'G',ha='center',va='center',size=20)

    if mark_pos == 'all':
        for x in range(size):
            for y in range(size):
                plt.text(x+0.5,y+0.25,policy[x][y],ha='center',va='center',size=20)
        plt.show(block=True)
    elif mark_pos != None:
        plt.text(mark_pos[0]+0.5,mark_pos[1]+0.25,action,ha='center',va='center',size=20)
    
    plt.show(block=False)
    time.sleep(0.5)
#    fig.clear()
    plt.close()

def postvalue(pos, action):
    p = [pos[0],pos[1]]
    if action == 'up': p[1] += 1
    elif action == 'down': p[1] -= 1
    elif action == 'left': p[0] -= 1
    elif action == 'right': p[0] += 1

    for i in [0,1]:
        if p[i] < 0: p[i] = 0
        if p[i] >= size: p[i] = size - 1

    return values[p[0]][p[1]]

def action_value(pos, action, goal):
    if pos == goal: return values[pos[0]][pos[1]]
    cur_v = values[pos[0]][pos[1]]
    post_v = postvalue(pos, action)
    swamp_cost = 0.0
    if pos == swamp: swamp_cost += 10.0

    return 1.0 + 0.9*post_v + 0.1*cur_v + swamp_cost #walk cost + 90% of transition + 10% of staying
		#return 0.9*(post_v + 1.0) + 0.1*(cur_v+1.0) # bellman equation
		
def sweep():
    changed = False
    for x in range(size):
        for y in range(size):
            best_value = 100
            best_action = None
            for a in actions:
                c = action_value((x,y), a, goal)
                if c < best_value:
                    best_value = c
                    best_action = a

            if math.fabs(values[x][y] - best_value) > 0.001:
                values[x][y] = best_value
                policy[x][y] = best_action
                draw((x,y), best_action)
                changed = True
    return changed

## TEST grid
#draw(None,None)

## TEST action
#print(action_value((2,1), 'up', goal))
#print(action_value((0,0), 'down', goal))

## TEST Iteration
draw(None, None)
changed = True
n = 1
while changed:
    print 'sweep'+str(n)
    changed = sweep()
    n = n + 1

draw("all",None)
