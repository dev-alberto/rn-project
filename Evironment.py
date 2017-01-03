import gym
import cv2


# "interface" for Environment, can be used to construct Gym Environment or own environment
class IEnvironment:
    def __init__(self):
        pass

    def num_actions(self):
        # Returns number of actions
        raise NotImplementedError

    def restart(self):
        # Restarts environment
        raise NotImplementedError

    def act(self, action):
        # Performs action and returns reward
        raise NotImplementedError

    def get_screen(self):
        # Gets current game screen
        raise NotImplementedError

    def is_terminal(self):
        # Returns if game is done
        raise NotImplementedError


class GymEnvironment(IEnvironment):
    def __init__(self):
        #IEnvironment.__init__(self)
        self.gym = gym.make('Breakout-v0')
        self.obs = None
        self.terminal = None

        self.screen_width = 84
        self.screen_height = 84

    def render(self):
        self.gym.render()

    def num_actions(self):
        return self.gym.action_space.n

    def restart(self):
        self.obs = self.gym.reset()
        self.terminal = False

    def act(self, action):
        self.obs, reward, self.terminal, _ = self.gym.step(action)
        return reward

    def get_screen(self):
        return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), (self.screen_width, self.screen_height))

    def is_terminal(self):
        return self.terminal
