import argparse

parser = argparse.ArgumentParser()

#environment args
env_args = parser.add_argument_group('Environment')

env_args.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
env_args.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")

#memory args
mem_args = parser.add_argument_group('Memory')

replay_size = 100000
#how many frames form a state
num_frames = 4



#q network args
q_args = parser.add_argument_group('QNetwork')

learning_rate = 0.0025
batch_size = 32
decay_rate = 0.95
min_reward = -1
max_reward = 1
optimizer = 'rmsprop'


#agent args
agent_args = parser.add_argument_group('Agent')

exploration_rate_start = 1
exploration_rate_end = 0.1
#Perform training after this many game steps.
train_frequency = 4

agent_args.append(exploration_rate_start)
agent_args.append(exploration_rate_end)
agent_args.append(train_frequency)
