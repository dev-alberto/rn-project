from __future__ import print_function
from Agent import Agent
from Evironment import GymEnvironment
from QNetwork import save_net
import numpy as np
import datetime
import sys


sys.setrecursionlimit(2000000000)
RANDOM_STEPS = 50000 #Populate replay memory with random steps before starting learning
EPOCHS = 50


def extract_epoch(net_file):
        return int(filter(str.isdigit, net_file))


def train_iteration(_agent, iteration):
        a = datetime.datetime.now().replace(microsecond=0)
        _agent.train()
        save_net(iteration)
        b = datetime.datetime.now().replace(microsecond=0)
        print("Completed " + str(iteration + 1) + "/" + str(EPOCHS) + " epochs in ", (b - a))


def resume_train(_agent, my_net):
        epoch = extract_epoch(my_net)
        _agent.play_random(RANDOM_STEPS)
        for i in range(epoch+1, EPOCHS):
                train_iteration(_agent, i)
        print("Training Session Ended.....")


def train(_agent):
        _agent.play_random(RANDOM_STEPS)
        for epoch in range(EPOCHS):
                train_iteration(_agent, epoch)
        print("Training Session Ended.....")

#rng = np.random.RandomState()

env = GymEnvironment()

train_agent = Agent(env)

resume_agent = Agent(env, start_epoch=18)

play_agent = Agent(env, p=True)

#train(train_agent)

#resume_train(resume_agent, 'saved_network/net-epoch17.pkl')

play_agent.play(10)
