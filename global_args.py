#environment args
env_args = []

screen_width = 84
screen_height = 84
display_screen = True

env_args.append(screen_height)
env_args.append(screen_width)
env_args.append(display_screen)

#memory args
mem_args = []

replay_size = 100000
#how many frames form a state
num_frames = 4

mem_args.append(replay_size)
mem_args.append(num_frames)

#q network args
q_args = []

learning_rate = 0.0025
batch_size = 32
decay_rate = 0.95
min_reward = -1
max_reward = 1
optimizer = 'rmsprop'

q_args.append(learning_rate)
q_args.append(batch_size)
q_args.append(decay_rate)
q_args.append(min_reward)
q_args.append(max_reward)
q_args.append(optimizer)

#agent args
agent_args = []

exploration_rate_start = 1
exploration_rate_end = 0.1
#Perform training after this many game steps.
train_frequency = 4

agent_args.append(exploration_rate_start)
agent_args.append(exploration_rate_end)
agent_args.append(train_frequency)
