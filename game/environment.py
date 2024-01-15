import numpy as np
import sys
import random
from itertools import cycle


S_range = [100, 140]
D_range = [60, 90]
signal_1D = np.random((1000))

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self, label_s, label_d, label_rad):
        self.score = self.playerIndex = self.loopIter = 0
        self.players = (S_range[0]+S_range[1])/2
        self.playerd = (D_range[0]+D_range[1])/2
        self.rad = max(S_range[1]-S_range[0], D_range[1]-D_range[0])/2
        self.label_s = label_s
        self.label_d = label_d
        self.label_rad = label_rad
        self.reward = [1, 0.2, -1]
        self.threshold = 0.5
        
    def step(self, input_actions, current_state):
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
            new_players = (self.players + S_range[0])/2
            new_playerd = (self.playerd + D_range[0])/2
            new_rad = self.rad/2
            
        if input_actions[1] == 1:
            new_players = (self.players + S_range[1])/2
            new_playerd = (self.playerd + D_range[0])/2
            new_rad = self.rad/2
            
        if input_actions[2] == 1:
            new_players = (self.players + S_range[0])/2
            new_playerd = (self.playerd + D_range[1])/2
            new_rad = self.rad/2
            
        if input_actions[3] == 1:
            new_players = (self.players + S_range[1])/2
            new_playerd = (self.playerd + D_range[1])/2
            new_rad = self.rad/2
        
        label_s_left = self.label_s-self.label_rad
        label_s_right = self.label_s-self.label_rad
        label_d_up = self.label_d-self.label_rad
        label_d_down = self.label_d+self.label_rad
        
        new_s_left = new_players - new_rad
        new_s_right = new_players + new_rad
        new_d_up = new_playerd - new_rad
        new_d_down = new_playerd + new_rad
        
        old_s_left = self.players-self.rad
        old_s_right = self.players+self.rad
        old_d_up = self.playerd-self.rad
        old_d_down = self.playerd+self.rad
        
        s_left = max(new_s_left, label_s_left)
        s_right = min(new_s_right, label_s_right)
        d_up = max(new_d_up, label_d_up)
        d_down = min(new_d_down, label_d_down)
        
        old_s_left = max(old_s_left, label_s_left)
        old_s_right = min(old_s_right, label_s_right)
        old_d_up = max(old_d_up, label_d_up)
        old_d_down = min(old_d_down, label_d_down)
        
        IOW_st_1_g = self.calcIOW(s_left, d_down, s_right, d_up, new_rad)
        IOW_st_g = self.calcIOW(old_s_left, old_d_down, old_s_right, old_d_up, self.rad)
        
        reward = None
        if IOW_st_1_g > self.threshold:
            reward = self.reward[0]
        elif IOW_st_1_g > IOW_st_g:
            reward = self.reward[1]
        else:
            reward = self.reward[2]
            
        if IOW_st_1_g == 0 or IOW_st_1_g == 1:
            done = True
        else: 
            done = False
        self.players = new_players
        self.playerd = new_playerd
        self.rad = new_rad
        new_state = [signal_1D, new_players, new_playerd, new_rad, current_state[-20:-4], input_actions]
        return 1, reward, done
    
    def calcIOW(s_left, d_down, s_right, d_up, newrad):
        if s_right < s_left or d_up < d_down:
            return 0.0
        inter_area = (s_right - s_left) * (d_up - d_down)
        IOW = inter_area / ((2 * newrad)**2)
        return IOW
    
    def checkdone():
        pass
