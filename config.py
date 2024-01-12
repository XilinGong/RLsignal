class Config():
    def __init__(self):
        self.memory_size = 100000
        self.max_episode = 30
        self.gamma = 0.90
        self.epsilon = 0.05
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.9
        self.learning_rate = -4
        self.var_init = 1.0
        self.dropout = 0.2
        self.layers = [512, 1024, 2048, 512]
        self.reward = [1, 0.2, -1]
        self.threshold = 0.5