from Agent import Agent
from Evironment import GymEnvironment
from Memory import Memory
from QNetwork import QNetwork
from global_args import env_args, mem_args, agent_args, q_args

env = GymEnvironment(env_args)
mem = Memory(mem_args)
q_net = QNetwork(q_args)
agent = Agent(env, mem, q_net, agent_args)


