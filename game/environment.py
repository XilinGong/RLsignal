import numpy as np
import sys
import random
from itertools import cycle


S_range = [85, 140]
D_range = [60, 100]


class GameState:
    def __init__(self, label_s, label_d, label_rad):
        self.score = self.playerIndex = self.loopIter = 0
        self.players = (S_range[0]+S_range[1])/2
        self.playerd = (D_range[0]+D_range[1])/2
        self.rad = max(S_range[1]-S_range[0], D_range[1]-D_range[0])/2
        self.label_s = label_s
        self.label_d = label_d
        self.label_rad = 2
        self.reward = [5, 1, -5]
        self.threshold = 0.5
        
    def step(self, input_actions, current_state, printing=False):
        if printing:
            print()
            print(input_actions)
            print('label', self.label_s, self.label_d)
            print('cur', self.players, self.playerd)
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        
        # S------------------>
        # D
        # |
        # |
        # |
        # V
    
        # input_actions[0] == 1: move left<--- up
        # input_actions[1] == 1: move right---> up
        # input_actions[2] == 1: move left<--- down
        # input_actions[3] == 1: move right---> down
        if input_actions[0] == 1:
            new_players = self.players - self.rad/2
            new_playerd = self.playerd - self.rad/2
            new_rad = self.rad/2
            
        if input_actions[1] == 1:
            new_players = self.players + self.rad/2
            new_playerd = self.playerd - self.rad/2
            new_rad = self.rad/2
            
        if input_actions[2] == 1:
            new_players = self.players - self.rad/2
            new_playerd = self.playerd + self.rad/2
            new_rad = self.rad/2
            
        if input_actions[3] == 1:
            new_players = self.players + self.rad/2
            new_playerd = self.playerd + self.rad/2
            new_rad = self.rad/2
        
        # label_s_left = self.label_s-self.label_rad
        # label_s_right = self.label_s-self.label_rad
        # label_d_up = self.label_d-self.label_rad
        # label_d_down = self.label_d+self.label_rad
        
        # new_s_left = new_players - new_rad
        # new_s_right = new_players + new_rad
        # new_d_up = new_playerd - new_rad
        # new_d_down = new_playerd + new_rad
        
        # old_s_left = self.players-self.rad
        # old_s_right = self.players+self.rad
        # old_d_up = self.playerd-self.rad
        # old_d_down = self.playerd+self.rad
        
        # s_left = max(new_s_left, label_s_left)
        # s_right = min(new_s_right, label_s_right)
        # d_up = max(new_d_up, label_d_up)
        # d_down = min(new_d_down, label_d_down)
        
        # old_s_left = max(old_s_left, label_s_left)
        # old_s_right = min(old_s_right, label_s_right)
        # old_d_up = max(old_d_up, label_d_up)
        # old_d_down = min(old_d_down, label_d_down)
        
        # IOW_st_1_g = self.calcIOW(s_left, d_down, s_right, d_up, new_rad)
        # IOW_st_g = self.calcIOW(old_s_left, old_d_down, old_s_right, old_d_up, self.rad)
        IOW_st_1_g = self.calcIOW2([new_players, new_playerd], new_rad, [self.label_s, self.label_d], self.label_rad)
        IOW_st_g = self.calcIOW2([self.players, self.playerd], self.rad, [self.label_s, self.label_d], self.label_rad)
        
        reward = None
        if IOW_st_1_g >= self.threshold:
            reward = self.reward[0]
        elif IOW_st_1_g > IOW_st_g:
            reward = self.reward[1]
        else:
            reward = self.reward[2]
            
        if reward == self.reward[2]:
            done = True
        elif  new_rad<= self.label_rad:
            done = True
        else:
            done = False
        self.players = new_players
        self.playerd = new_playerd
        self.rad = new_rad
        signal = current_state[:1000]
        new_state = np.concatenate((signal, np.array([new_players, new_playerd, new_rad]),current_state[-16:],np.array(input_actions)))
        return new_state, reward, done
    
    def calcIOW2(self, sd1, r1, sd2, r2):
        s1, d1 = sd1
        s2, d2 = sd2
        area_fm = 4*r1**2
        s_max = min(s1+r1, s2+r2)
        s_min = max(s1-r1, s2-r2)
        if s_max < s_min:
            return 0.0
        d_max = min(d1+r1, d2+r2)
        d_min = max(d1-r1, d2-r2)
        if d_max < d_min:
            return 0.0
        inter_area = (s_max-s_min)*(d_max-d_min)
        iow = inter_area/area_fm
        return iow
    
        
    def calcIOW(self, s_left, d_down, s_right, d_up, newrad): 
        if s_right < s_left or d_up < d_down:
            return 0.0
        inter_area = (s_right - s_left) * (d_up - d_down)
        IOW = inter_area / ((2 * newrad)**2)
        return IOW
