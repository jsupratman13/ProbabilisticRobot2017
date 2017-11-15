import random, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Agent(object):
    def __init__(self):
        self.actions = ["up", "down", "left", "right"]
        self.pos = [0,0]

class State(object):
    def __init__(self, actions):
        self.Q ={}
        for a in actions:
            self.Q[a] = 0.0
        self.best_action = "up"
        self.goal = False

    def set_goal(self, actions):
        for a in actions:
            self.Q[a] = 0.0
        self.goal = True

def draw(mark_pos):
    fig, ax = plt.subplots()
    values = [[states[i][j].Q[states[i][j].best_action]for j in range(size)]for i in range(size)]
    mp = ax.pcolor(values, cmap=plt.cm.YlOrRd, vmin=0, vmax=8)
    ax.set_aspect(1)
    ax.set_xticks(range(size), minor=False)
    ax.set_yticks(range(size), minor=False)

    for x in range(len(values)):
        for y in range(len(values[0])):
            s = states[x][y]
            plt.text(x+0.5,y+0.5,round(s.Q[s.best_action],2),ha='center',va='center',size=20)

            if states[x][y].goal:
                plt.text(x+0.75,y+0.75,'G',ha='center',va='center',size=20)

    plt.text(agent.pos[0]+0.5, agent.pos[1]+0.25,'agent',ha='center',va='center',size=20)

    if mark_pos == 'all':
        for x in range(size):
            for y in range(size):
                if states[x][y].goal: continue
                plt.text(x+0.5,y+0.25,states[x][y].best_action,ha='center',va='center',size=20)
    elif mark_pos != None:
        s = states[mark_pos[0]][mark_pos[1]]
        plt.text(mark_pos[0]+0.5,mark_pos[1]+0.25,s.best_action,ha='center',va='center',size=20)
   
    plt.show(block=False)
    time.sleep(0.5)
    #fig.clear()
    #plt.pause(0.5)
    fig.clear()
    plt.close()

def state_transition(s_pos, a):
    #10% probability of failing to move
    if random.uniform(0,1) < 0.1:
        return s_pos

    x,y = s_pos
    if a == 'up': y+=1
    elif a == 'down': y-=1
    elif a == 'right': x += 1
    elif a == 'left' : x -= 1

    if x < 0: x = 0
    elif x >= size: x = size - 1
    if y < 0: y = 0
    elif y >= size: y = size -1

    return [x,y]

def e_greedy(s):
    if random.uniform(0,1) < 0.1:
        return random.choice(agent.actions)
    else:
        best_a = None
        best_q = 10000000
        for a in s.Q:
            if best_q > s.Q[a]:
                best_q = s.Q[a]
                best_a = a
        s.best_action = best_a
        return best_a

def sarsa(s_pos,a):
    alpha = 0.5
    gamma = 1.0

    s = states[s_pos[0]][s_pos[1]]
    s_next_pos = state_transition(s_pos, a)
    s2 = states[s_next_pos[0]][s_next_pos[1]]
    a2 = e_greedy(s2)

    q = (1.0-alpha)*s.Q[a]+alpha*(1.0+gamma*s2.Q[a2])
    print 's: ' + str(s_pos) + ' a: ' + str(a) + ' s2: ' + str(s_next_pos) + ' a2: ' + str(a2)
    print '----------------------'
    return s_next_pos, a2, q

def one_step():
    agent.pos = [random.randrange(size), random.randrange(size)]
    a = e_greedy(states[agent.pos[0]][agent.pos[1]])
    if states[agent.pos[0]][agent.pos[1]].goal:
        return

    while True:
        #draw(None)
        s2, a2, q = sarsa(agent.pos, a)
        states[agent.pos[0]][agent.pos[1]].Q[a] = q
        agent.pos = s2
        a = a2
        if states[agent.pos[0]][agent.pos[1]].goal:
            break

def train():
    one_step()
    draw('all')

agent = Agent()
size = 3
states = [[State(agent.actions) for i in range(size)] for j in range(size)]
states[2][2].set_goal(agent.actions)

#draw(None)
for i in range(3):
    train()
