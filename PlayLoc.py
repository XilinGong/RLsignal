from DQNAgent import DQNAgent
import numpy as np
from config import Config

def calcIOW(s_left, d_down, s_right, d_up, newrad):
    if s_right < s_left or d_up < d_down:
        return 0.0
    inter_area = (s_right - s_left) * (d_up - d_down)
    IOW = inter_area / ((2 * newrad)**2)
    return IOW

def step(state, action, label, cfg = Config()):
    s, d, rad = state[1000], state[1001], state[1002]
    
    if action == 0:
        new_s = (s + d)/2
        onehot_action = [1, 0]
    else:
        new_d = (s + d)/2
        onehot_action = [0, 1]
    new_rad = 1/2 * rad
    old_state = state
    state[1000], state[1001] = new_s, new_d
    state[1002] = new_rad
    state[1003:-2] = state[1005:]
    state[: -2] = onehot_action
    new_state = state
    labels, labeld, labelrad = label
    new_s_lw = new_s - new_rad
    new_d_lw = new_d - new_rad
    new_s_ru = new_s + new_rad
    new_d_ru = new_d + new_rad
    old_s_lw = s - rad
    old_d_lw = d - rad
    old_s_ru = s + rad
    old_d_ru = d + rad
    label_s_lw = labels - labelrad
    label_d_lw = labeld - labelrad
    label_s_ru = labels + labelrad
    label_d_ru = labeld + labelrad
    s_left = max(new_s_lw, label_s_lw)
    d_down = max(label_d_lw, new_d_lw)
    s_right = min(new_s_ru, label_s_ru)
    d_up = min(label_d_ru, new_d_ru)
    
    old_s_left = max(old_s_lw, label_s_lw)
    old_d_down = max(label_d_lw, old_d_lw)
    old_s_right = min(old_s_ru, label_s_ru)
    old_d_up = min(label_d_ru, old_d_ru)
    IOW_st_1_g = calcIOW(s_left, d_down, s_right, d_up, new_rad)
    IOW_st_g = calcIOW(old_s_left, old_d_down, old_s_right, old_d_up, rad)
    reward = None
    if IOW_st_1_g > cfg.threshold:
        reward = cfg.reward[0]
    elif IOW_st_1_g > IOW_st_g:
        reward = cfg.reward[1]
    else:
        reward = cfg.reward[2]
    if IOW_st_1_g == 0 or IOW_st_1_g == 1:
        done = True
    else: 
        done = False
            
    return old_state, action, reward, new_state, done
        
    
    

def playLoc():
    # init agent
    label = [15, 105, 1]
    actions = 2
    state_size = 1000 + 2 + 1 + 2 * 20
    agent = DQNAgent(state_size=state_size, action_size=actions)
    done = False
    srange = [10,100]
    drange = [100,200]
    signal = np.random.rand(1, 1000)
    state = np.zeros((1, state_size))
    state[0:1000] = signal
    state[1001] = (srange[0]+srange[1])/2
    state[1002] = (drange[0]+drange[1])/2
    state[1003] = max(srange[1]-srange[0], drange[1]-drange[0])/2
    
    while not done:
        action0 = agent.choose_action(state)
        oldstate, action, reward, new_state, done = step(state, action0, label)
        agent.store_transition(oldstate, action0, reward, new_state, done)
    
    