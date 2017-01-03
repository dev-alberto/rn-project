from __future__ import print_function
from Agent import Agent
from Evironment import GymEnvironment
from Memory import Memory
from QNetwork import QNetwork
import numpy as np
import datetime


STEPS_PER_EPOCH = 500
STEPS_PER_TEST = 100
EPOCHS = 3

rng = np.random.RandomState()
env = GymEnvironment()
mem = Memory()
q_net = QNetwork(env.num_actions(), np.random.RandomState(123456))
agent = Agent(env, mem, q_net)

agent.play_random(random_steps=100)
print("Traning Started.....")
for i in range(EPOCHS):
        #stats.reset()
        a = datetime.datetime.now().replace(microsecond=0)
        agent.train(train_steps = STEPS_PER_EPOCH,epoch = 1)
        b = datetime.datetime.now().replace(microsecond=0)
        print("Completed " + str(i + 1) + "/" + str(EPOCHS) + " epochs in ", (b - a))

print("Training Ended.....")

#agent.play(1)
