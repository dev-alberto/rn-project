import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import dnn


class QNetwork:
    def __init__(self, num_actions, rng, screen_height=84, screen_width=84, discount_rate=.99, batch_size=32,
                 history_length=4, rms_decay=.95, learning_rate=.00025, rms_epsilon=1e-06, clip_error=1,
                 input_scale=255):
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.discount_rate = discount_rate  # discount factor gamma used in the Q learning update
        self.batch_size = batch_size
        self.history_length = history_length  # The number of most recent frames experienced by the agent
        self.rms_decay = rms_decay
        self.learning_rate = learning_rate
        self.rms_epsilon = rms_epsilon
        self.clip_error = clip_error
        self.rng = rng

        lasagne.random.set_rng(self.rng)  # set the seed

        self.update_counter = 0

        self.net = self.build_network(num_actions)

        # 4-dimensional ndarray (similar to prestates in memory_store)
        states = T.tensor4('states')
        # 4-dimensional ndarray (similar to poststates in memory_store)
        post_states = T.tensor4('post_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        # creating a shared object is like declaring global - it has be shared between functions that it appears in.
        # similar to prestates matrix construction in memory_store
        self.states_shared = theano.shared(
            np.zeros((batch_size, history_length, screen_height, screen_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, history_length, screen_height, screen_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # compute an expression for the output of a single layer given its input
        # scaling turns grayscale (or) black and white to 1s and 0s (black OR white)
        q_vals = lasagne.layers.get_output(self.net, states / input_scale)
        next_q_vals = lasagne.layers.get_output(self.net, post_states / input_scale)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        # perform this step: Q(st,a) = rimm + gamma*[ max(a{t+1}) Q(s{t+1}, a{t+1})]
        # col. of ones with same dim. as terminals
        target = (rewards + (T.ones_like(terminals) - terminals) * self.discount_rate * T.max(next_q_vals,
                                                                                              axis=1, keepdims=True))

        # col. matrix into row matrix|row. matrix into col matrix
        diff = target - q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))

        quadratic_part = T.minimum(abs(diff), self.clip_error)
        linear_part = abs(diff) - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_error * linear_part
        loss = T.mean(loss)

        params = lasagne.layers.helper.get_all_params(self.net)
        givens = {
            states: self.states_shared,
            post_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        updates = lasagne.updates.rmsprop(loss, params, self.learning_rate, self.rms_decay, self.rms_epsilon)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})

    def build_network(self, num_actions):
        input_layer = lasagne.layers.InputLayer(shape=(self.batch_size, self.history_length,
                                                       self.screen_height, self.screen_width))

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
        convolutional_layer1 = dnn.Conv2DDNNLayer(input_layer, num_filters=32, filter_size=(8, 8), stride=(4, 4),
                                                  nonlinearity=lasagne.nonlinearities.rectify,
                                                  W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

        convolution_layer2 = dnn.Conv2DDNNLayer(convolutional_layer1, num_filters=64, filter_size=(4, 4), stride=(2, 2),
                                                nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeUniform(),
                                                b=lasagne.init.Constant(.1))

        convolution_layer3 = dnn.Conv2DDNNLayer(convolution_layer2, num_filters=64, filter_size=(3, 3), stride=(1, 1),
                                                nonlinearity=lasagne.nonlinearities.rectify,
                                                W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

        hidden_layer1 = lasagne.layers.DenseLayer(convolution_layer3, num_units=512,
                                                  nonlinearity=lasagne.nonlinearities.rectify,
                                                  W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))

        output_layer = lasagne.layers.DenseLayer(hidden_layer1, num_units=num_actions, nonlinearity=None,
                                                 W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
        return output_layer

    def train(self, states, actions, rewards, next_states, terminals):
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(np.matrix(actions).T)
        self.rewards_shared.set_value(np.matrix(rewards, dtype=theano.config.floatX).T)
        self.terminals_shared.set_value(np.matrix(terminals).T)
        loss, _ = self._train()
        # print loss,_
        self.update_counter += 1
        # print "finished train step.sqrt of loss: ",np.sqrt(loss)
        return np.sqrt(loss)

    def q_vals(self, state):
        states = np.zeros((self.batch_size, self.history_length, self.screen_height,
                           self.screen_width), dtype=theano.config.floatX)
        states[0, Ellipsis] = state
        self.states_shared.set_value(states)
        return self._q_vals()[0]

    def choose_action(self, state):
        # print "predicting"
        q_vals = self.q_vals(state)
        # print "what to choose? ",q_vals
        return np.argmax(q_vals)

    def predict(self, state):
        q_vals = self.q_vals(state)
        return q_vals
