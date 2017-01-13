from __future__ import print_function
import random
import datetime
from Memory import add, get_minibatch, get_current_state
from QNetwork import train, choose_action


class Agent:
    def __init__(self, environment,
                 start_epoch=0, p=False, train_steps=250000, random_starts=30, history_length=4,
                 exploration_rate_start=1, exploration_rate_end=0.1, exploration_decay_steps=1000000,
                 train_frequency=4, train_repeat=1):

        self.env = environment


        self.num_actions = self.env.num_actions()


        self.random_starts = random_starts
        self.history_length = history_length

        self.exploration_rate_start = exploration_rate_start
        self.exploration_rate_end = exploration_rate_end

        #How many steps to decay the exploration rate.
        self.exploration_decay_steps = exploration_decay_steps

        self.train_frequency = train_frequency

        #Number of times to sample minibatch during training
        self.train_repeat = train_repeat

        #Perform this many steps per epoch
        self.train_steps = train_steps

        #model total train time
        self.total_train_steps = start_epoch * train_steps

        self.total_score = 0.0
        self.total_moves = 0

        self.p = p

    def restart_random(self):
        self.env.restart()
        # perform random number of dummy actions to produce more stochastic games
        for i in range(random.randint(self.history_length, self.random_starts) + 1):
            reward = self.env.act(0)
            screen = self.env.get_screen()
            terminal = self.env.is_terminal()
            assert not terminal, "terminal state occurred during random initialization"
            # add dummy states to buffer to guarantee history_length screens
            add(0, reward, screen, terminal)

    def _exploration_rate(self):
        # calculate decaying exploration rate - very slowly slowly go from 0.1 to 1.0 over decay_steps steps
        # exploration rate controls when to use the model to predict and when to randomly select a action
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (
                self.exploration_rate_start - self.exploration_rate_end) / float(self.exploration_decay_steps)
        else:
            return self.exploration_rate_end

    def step(self, exploration_rate):
        # perform a single step (a single action)
        if self.p:
            """Display game screen if mode is set to play"""
            self.env.render()
        self.total_moves += 1
        if random.random() < exploration_rate:
            # randomly select a action based on exploration_rate
            action = random.randrange(self.num_actions)
        else:
            # choose action with highest Q-value
            state = get_current_state()
            # for convenience getCurrentState() returns minibatch
            action = choose_action(state)
            # print "Prediction chosen",action
          #  moves_print = " Exploration rate:	" + str(exploration_rate) + " action is: " + str(action)
            #moves_chosen = open("moves.txt", "a")
           # moves_chosen.write(moves_print+"\n")
           # moves_chosen.close()
        reward = self.env.act(action)
        screen = self.env.get_screen()
        terminal = self.env.is_terminal()

        self.total_score += reward

        add(action, reward, screen, terminal)

        # if game over, then restart it
        if terminal:
            a = datetime.datetime.now().replace(microsecond=0)
            print_stats = "Time: " + str(a) + "	Total moves made in a game: " + str(
                self.total_moves) + "			Total Score: " + str(self.total_score)
            # print print_stats
            log = open("log_stuff.txt", "a")
            log.write(print_stats + "\n")
            log.close()
            self.total_score = 0
            self.total_moves = 0
            self.restart_random()
        return terminal

    def play_random(self, random_steps):
        # randomly play for given no. of steps
        for i in range(random_steps):
            # use exploration rate 1 = completely random
            self.step(1)
            #print("random step", i)

    def train(self):
        # play given number of steps
        for i in range(self.train_steps):
            # perform a single game step
            self.step(self._exploration_rate())
            # train after every train_frequency (4)	steps
            if i % self.train_frequency == 0:
                # train for train_repeat times
                for j in range(self.train_repeat):
                    # sample minibatch
                    minibatch = get_minibatch()
                    train(minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4])
            self.total_train_steps += 1

    def play(self, num_games):
        self.restart_random()
        for i in range(num_games):
            # play until terminal state, with 0.05 exploration rate, so no random allowed :))
            while not self.step(0.05):
                pass
