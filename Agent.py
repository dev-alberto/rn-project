class Agent:
    def __init__(self, environment, memory, q_network, args):
        self.env = environment
        self.mem = memory
        self.net = q_network

    def train(self):
        pass

    def play(self):
        pass
