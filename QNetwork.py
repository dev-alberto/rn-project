from __future__ import print_function
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import dnn
import cPickle

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
number_env_actions = 6  # atari setup
input_scale = 255


lasagne.random.set_rng(rng)  # set the seed

# creating a shared object, declaring it global - it has be shared between functions that it appears in.
# similar to pre_states matrix construction in memory
states_shared = theano.shared(np.zeros((mini_batch_size, history_length, screen_height, screen_width),
                                       dtype=theano.config.floatX))

post_states_shared = theano.shared(np.zeros((mini_batch_size, history_length, screen_height, screen_width),
                                            dtype=theano.config.floatX))

rewards_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype=theano.config.floatX),
                               broadcastable=(False, True))

actions_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype='int32'),
                               broadcastable=(False, True))

terminals_shared = theano.shared(np.zeros((mini_batch_size, 1), dtype='int32'),
                                 broadcastable=(False, True))
# 4-dimensional ndarray (similar to prestates in memory_store)
states = T.tensor4('states')
# 4-dimensional ndarray (similar to poststates in memory_store)
post_states = T.tensor4('post_states')
rewards = T.col('rewards')
actions = T.icol('actions')
terminals = T.icol('terminals')

givens = {
        states: states_shared,
        post_states: post_states_shared,
        rewards: rewards_shared,
        actions: actions_shared,
        terminals: terminals_shared
    }


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

    convolution_layer1 = dnn.Conv2DDNNLayer(input_layer, num_filters=16, filter_size=(8, 8), stride=(4, 4),
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.Normal(.01), b=lasagne.init.Constant(.1))

    convolution_layer2 = dnn.Conv2DDNNLayer(convolution_layer1, num_filters=32, filter_size=(4, 4), stride=(2, 2),
                                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Normal(.01),
                                            b=lasagne.init.Constant(.1))

    hidden_layer1 = lasagne.layers.DenseLayer(convolution_layer2, num_units=256,
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.Normal(.01), b=lasagne.init.Constant(.1))

    output_layer = lasagne.layers.DenseLayer(hidden_layer1, num_units=number_env_actions, nonlinearity=None,
                                             W=lasagne.init.Normal(.01), b=lasagne.init.Constant(.1))

    return output_layer

#net = build_net()


def load_model(_file):
    """Comment the above line and set net variable to: net = load_model('my_model')"""
    print("Model Loaded")
    return cPickle.load(open(_file, 'r'))

net = load_model('saved_network/net-epoch29.pkl')


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
    # If we simply take the squared clipped diff as our loss,
    # then the gradient will be zero whenever the diff exceeds
    # the clip bounds. To avoid this, we extend the loss
    # linearly past the clip point to keep the gradient constant
    # This is equivalent to declaring d loss/d q_vals to be
    # equal to the clipped diff, then backpropagating from
    # there, which is what the DeepMind implementation does.
    quadratic_part = T.minimum(abs(difference), clip_error)
    linear_part = abs(difference) - quadratic_part
    _loss = 0.5 * quadratic_part ** 2 + clip_error * linear_part
    return T.mean(_loss)


def mini_batch_optimisation(_loss, _net_params):
    """Normalize gradient"""
    return lasagne.updates.rmsprop(_loss, _net_params, learning_rate, rms_decay, rms_epsilon)


#def compute_theano_funct_parameters():
__q_values = get_layer_output(net, states)
next_q_values = get_layer_output(net, post_states)
next_q_values = theano.gradient.disconnected_grad(next_q_values)

__loss = get_loss(rewards, terminals, __q_values, next_q_values, actions)

_net_parameters = get_network_parameters(net)

"""updates : describe how to update the shared value"""
_updates = mini_batch_optimisation(__loss, _net_parameters)

    #print("debug")
#return _updates, _loss, __q_values
    # return theano.function([], [_loss, q_values], updates=_updates, givens=_givens)


# print(make_theano_loss_function()[1])

#params = compute_theano_funct_parameters()
#print(params)
'''Symbolic theano functions for our loss and q-values'''
f1 = theano.function([], [__loss, __q_values], updates=_updates, givens=givens)
f2 = theano.function([], __q_values, givens={states: states_shared})


def train(_states, _actions, _rewards, _next_states, _terminals):
    states_shared.set_value(_states)
    post_states_shared.set_value(_next_states)
    actions_shared.set_value(np.matrix(_actions).T)
    rewards_shared.set_value(np.matrix(_rewards, dtype=theano.config.floatX).T)
    terminals_shared.set_value(np.matrix(_terminals).T)
    #caca = compute_theano_funct_parameters()
    _loss, _ = f1()
    #print(_loss)
    # print("finished train step.sqrt of loss: ", np.sqrt(_loss))
    return np.sqrt(_loss)


def q_vals(_state):
    _states = np.zeros((mini_batch_size, history_length, screen_height,
                        screen_width), dtype=theano.config.floatX)
    """ Ellipsis is used here to indicate a placeholder for the rest of the array, cool slicing """
    _states[0, Ellipsis] = _state
    states_shared.set_value(_states)
    #caca = compute_theano_funct_parameters()
    # q_values = get_layer_output(net, _states)
    return f2()[0]


def choose_action(_state):
    """Given state, make choice with highest q value"""
    # print "predicting"
    q_values = q_vals(_state)
    # print "what to choose? ",q_vals
    return np.argmax(q_values)


def save_net(iteration):
    path = 'saved_network' + '/net' + '-epoch' + str(iteration) + '.pkl'
    cPickle.dump(net, open(path, 'wb'), cPickle.HIGHEST_PROTOCOL)
