from __future__ import print_function
from Agent import Agent
from Evironment import GymEnvironment
from Memory import Memory
from QNetwork import QNetwork
import numpy as np
import datetime
import cPickle
import sys
import json

sys.setrecursionlimit(2000000000)
STEPS_PER_EPOCH = 50000
STEPS_PER_TEST = 10000
RANDOM_STEPS = 2500
EPOCHS = 20


def extract_epoch(net_file):
        return int(filter(str.isdigit, net_file))


def load_net(_file):
        return cPickle.load(open(_file, 'r'))

def train_iteration(_agent, iteration):
        a = datetime.datetime.now().replace(microsecond=0)
        _agent.train(train_steps=STEPS_PER_EPOCH, epoch=1)
        path = 'saved_network' + '/rmspropNet' + '-epoch' + str(iteration) + '.pkl'
        cPickle.dump(_agent.get_net(), open(path, 'wb'), cPickle.HIGHEST_PROTOCOL)
        b = datetime.datetime.now().replace(microsecond=0)
        print("Completed " + str(iteration + 1) + "/" + str(EPOCHS) + " epochs in ", (b - a))


def resume_train(_agent, _file):
        epoch = extract_epoch(_file)
        _agent.set_net(load_net(_file))
        _agent.play_random(RANDOM_STEPS)
        for i in range(epoch+1, EPOCHS):
                train_iteration(_agent, i)
        print("Training Session Ended.....")


def train(_agent):
        _agent.play_random(RANDOM_STEPS)
        for i in range(EPOCHS):
                train_iteration(_agent, i)
        print("Training Session Ended.....")

rng = np.random.RandomState()
env = GymEnvironment()
mem = Memory()
q_net = QNetwork(env.num_actions(), np.random.RandomState(123456))
#q_net = load_net('saved_network/rmspropNet-epoch4.pkl')
agent = Agent(env, mem, q_net)

#train(agent)

resume_train(agent, 'saved_network/rmspropNet-epoch5.pkl')

#agent.play(1)
