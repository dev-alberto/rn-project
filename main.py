from __future__ import print_function
from Agent import Agent
from Evironment import GymEnvironment
from QNetwork import net
import numpy as np
import datetime
import cPickle
import sys

sys.setrecursionlimit(2000000000)
RANDOM_STEPS = 300 #Populate replay memory with random steps before starting learning
EPOCHS = 2


def extract_epoch(net_file):
        return int(filter(str.isdigit, net_file))


def load_net(_file):
        return cPickle.load(open(_file, 'r'))


def train_iteration(_agent, iteration):
        a = datetime.datetime.now().replace(microsecond=0)
        _agent.train()
        path = 'saved_network' + '/net' + '-epoch' + str(iteration) + '.pkl'
        cPickle.dump(net, open(path, 'wb'), cPickle.HIGHEST_PROTOCOL)
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
        for epoch in range(EPOCHS):
                train_iteration(_agent, epoch)
        print("Training Session Ended.....")

#rng = np.random.RandomState()
env = GymEnvironment()
#q_net = QNetwork(env.num_actions(), np.random.RandomState(123456))
#q_net = load_net('saved_network/rmspropNet-epoch24.pkl')

start_agent = Agent(env)
#resume_agent = Agent(env, q_net, start_epoch=26)

train(start_agent)

#resume_train(resume_agent, 'saved_network/net-epoch25.pkl')

#agent.play(1)
