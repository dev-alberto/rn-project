from __future__ import print_function
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import dnn

# Network Hyperparameters, as described in the Google DeepMind
# Paper : http://files.davidqiu.com/research/nature14236.pdf

# discount factor gamma used in the Q learning update
discount_rate = 0.99

# Number of training cases over which each stochastic update is computed
mini_batch_size = 32

# The number of most recent frames experienced by the agent
history_length = 4

# Decay rate for RMSprop
rms_decay = .95

# Learning rate for RMSprop
learning_rate = .00025

# Epsilon value for RMSprop
rms_epsilon = 1e-6

# Clip error term in update between this number and its negative
clip_error = 1

# Random seeds
rng = np.random.RandomState(123456)

# Environment parameters
screen_height = 84
screen_width = 84
number_env_actions = 18  # atari setup
input_scale = 255

update_counter = 0

lasagne.random.set_rng(rng)  # set the seed


# creating a shared object is like declaring global - it has be shared between functions that it appears in.
# similar to pre_states matrix construction in memory_store
states_shared = theano.shared(np.zeros((mini_batch_size, history_length, screen_height, screen_width),
                                       dtype=theano.config.floatX))

next_states_shared = theano.shared(np.zeros((mini_batch_size, history_length, screen_height, screen_width),
                                            dtype=theano.config.floatX))

rewards_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype=theano.config.floatX),
                               broadcastable=(False, True))

actions_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype='int32'),
                               broadcastable=(False, True))

terminals_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype='int32'),
                                 broadcastable=(False, True))


def build_net():
    """Build deeep q network, exactly as described in the deep mind paper  """
    input_layer = lasagne.layers.InputLayer(shape=(mini_batch_size, history_length,
                                                   screen_height, screen_width))

    # First attempt at building the network, very slow CPU training

    # l_conv1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(8, 8), stride=(4, 4),
    #                                      nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeUniform(),
    #                                      b=lasagne.init.Constant(.1))
    #
    # l_conv2 = lasagne.layers.Conv2DLayer(l_conv1, num_filters=64, filter_size=(4, 4), stride=(2, 2),
    #                                      nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeUniform(),
    #                                      b=lasagne.init.Constant(.1))
    #
    # l_conv3 = lasagne.layers.Conv2DLayer(l_conv2, num_filters=64, filter_size=(3, 3), stride=(1, 1),
    #                                      nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeUniform(),
    #                                      b=lasagne.init.Constant(.1))
    #
    # l_hidden1 = lasagne.layers.DenseLayer(l_conv3, num_units=512, nonlinearity=lasagne.nonlinearities.rectify,
    #                                       W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
    #
    # l_out = lasagne.layers.DenseLayer(l_hidden1, num_units=num_actions, nonlinearity=None,
    #                                   W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

    # Second network, utilisez NVIDIA GPU for training

    convolution_layer1 = dnn.Conv2DDNNLayer(input_layer, num_filters=32, filter_size=(8, 8), stride=(4, 4),
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

    convolution_layer2 = dnn.Conv2DDNNLayer(convolution_layer1, num_filters=64, filter_size=(4, 4), stride=(2, 2),
                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeUniform(),
                                            b=lasagne.init.Constant(.1))

    convolution_layer3 = dnn.Conv2DDNNLayer(convolution_layer2, num_filters=64, filter_size=(3, 3), stride=(1, 1),
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

    hidden_layer1 = lasagne.layers.DenseLayer(convolution_layer3, num_units=512,
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

    output_layer = lasagne.layers.DenseLayer(hidden_layer1, num_units=number_env_actions, nonlinearity=None,
                                             W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
    return output_layer

net = build_net()
print("net created")


def get_layer_output(_net, _states):
    """compute an expression for the output of a single layer given its input"""
    return lasagne.layers.get_output(_net, _states / input_scale)


def get_network_parameters(_net):
    return lasagne.layers.helper.get_all_params(_net)


def get_loss(_rewards, _terminals, _q_vals, _next_q_vals, _actions):
    # perform this step: Q(st,a) = rewards + gamma*[ max(a{t+1}) Q(s{t+1}, a{t+1})]
    target = (_rewards + (T.ones_like(_terminals) - _terminals) * discount_rate * T.max(_next_q_vals,
                                                                                        axis=1, keepdims=True))
    # col. matrix into row matrix|row. matrix into col matrix
    difference = target - _q_vals[T.arange(mini_batch_size), _actions.reshape((-1,))].reshape((-1, 1))
    quadratic_part = T.minimum(abs(difference), clip_error)
    linear_part = abs(difference) - quadratic_part
    _loss = 0.5 * quadratic_part ** 2 + clip_error * linear_part
    return T.mean(_loss)


def mini_batch_optimisation(_loss, _net_params):
    return lasagne.updates.rmsprop(_loss, _net_params, learning_rate, rms_decay, rms_epsilon)


def compute_theano_funct_parameters():
    # 4-dimensional ndarray (similar to prestates in memory_store)
    states = T.tensor4('states')
    # 4-dimensional ndarray (similar to poststates in memory_store)
    post_states = T.tensor4('post_states')
    rewards = T.col('rewards')
    actions = T.icol('actions')
    terminals = T.icol('terminals')

    q_values = get_layer_output(net, states)
    next_q_values = get_layer_output(net, post_states)
    next_q_values = theano.gradient.disconnected_grad(next_q_values)

    _loss = get_loss(rewards, terminals, q_values, next_q_values, actions)

    _net_parameters = get_network_parameters(net)

    _givens = {
        states: states_shared,
        post_states: next_states_shared,
        rewards: rewards_shared,
        actions: actions_shared,
        terminals: terminals_shared
    }

    _updates = mini_batch_optimisation(_loss, _net_parameters)

    return _givens, _updates, _loss, q_values, states
    #print("debug")
    #return theano.function([], [_loss, q_values], updates=_updates, givens=_givens)

#print(make_theano_loss_function()[1])

params = compute_theano_funct_parameters()

f1 = theano.function([], [params[2], params[3]], updates=params[1], givens=params[0])
f2 = theano.function([], params[3], givens={params[4]: states_shared})


def train(_states, _actions, _rewards, _next_states, _terminals):
    global update_counter
    states_shared.set_value(_states)
    next_states_shared.set_value(_next_states)
    actions_shared.set_value(np.matrix(_actions).T)
    rewards_shared.set_value(np.matrix(_rewards, dtype=theano.config.floatX).T)
    terminals_shared.set_value(np.matrix(_terminals).T)
    _loss, _ = f1()
    # print loss,_
    update_counter += 1
    #print("finished train step.sqrt of loss: ", np.sqrt(_loss))
    return np.sqrt(_loss)


def q_vals(_state):
    _states = np.zeros((mini_batch_size, history_length, screen_height,
                        screen_width), dtype=theano.config.floatX)
    _states[0, Ellipsis] = _state
    states_shared.set_value(_states)
    #q_values = get_layer_output(net, _states)
    return f1()[0]


def choose_action(_state):
    """Given state, make choice with highest q value"""
    # print "predicting"
    q_values = q_vals(_state)
    # print "what to choose? ",q_vals
    return np.argmax(q_values)
