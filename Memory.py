import numpy as np
import random

#constants
size = 1000000
history_length = 4
screen_height = 84
screen_width = 84
batch_size = 32

# preallocate memory
actions = np.empty(size, dtype=np.uint8)
rewards = np.empty(size, dtype=np.int64)
terminals = np.empty(size, dtype=np.bool)
history_length = history_length
dims = (screen_height, screen_width)
batch_size = batch_size
min_reward = -1.0
max_reward = 1.0
count = 0
current = 0
screens = np.empty((size, screen_height, screen_width), dtype=np.uint8)

# pre-allocate prestates and poststates for minibatch
pre_states = np.empty((batch_size, history_length) + dims, dtype=np.uint8)
post_states = np.empty((batch_size, history_length) + dims, dtype=np.uint8)


def add(action, reward, screen, terminal):
    global current, count
    actions[current] = action
    # clip reward between -1 and 1
    if min_reward and reward < min_reward:
        reward = max(reward, min_reward)
    if max_reward and reward > max_reward:
        reward = min(reward, max_reward)
    rewards[current] = reward
    # screen is 84x84 size
    screens[current, Ellipsis] = screen
    terminals[current] = terminal
    count = max(count, current + 1)
    current = (current + 1) % size


def get_state(index):

    index %= count

    if index >= history_length - 1:
        return screens[(index - (history_length - 1)):(index + 1), Ellipsis]
    else:
        indexes = [(index - i) % count for i in reversed(range(history_length))]

        return screens[indexes, Ellipsis]


def get_current_state():
    # this is the input to the model to predict what move to make next
    # reuse first row of prestates in minibatch to minimize memory consumption
    pre_states[0, Ellipsis] = get_state(current - 1)
    # print self.getState(self.current - 1).shape, "is shape of getstate"
    # print self.prestates.shape,"is the shape of current state"
    current_state = get_state(current - 1)
    return current_state


def get_minibatch():
    # sample random indexes
    indexes = []
    while len(indexes) < batch_size:
        while True:
            index = random.randint(history_length, count - 1)
            if index >= current > index - history_length:
                continue
            if terminals[(index - history_length):index].any():
                continue
            # otherwise use this index
            break

        # fill the "batch"
        pre_states[len(indexes), Ellipsis] = get_state(index - 1)
        post_states[len(indexes), Ellipsis] = get_state(index)
        indexes.append(index)

    # copy actions, rewards and terminals directly
    _actions = actions[indexes]
    _rewards = rewards[indexes]
    _terminals = terminals[indexes]
    return pre_states, _actions, _rewards, post_states, _terminals
