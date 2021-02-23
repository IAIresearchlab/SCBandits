import numpy as np
import random
from base_model import *
from output_format import *


class Greedy(BaseModel):
    '''
    Greedy model for Epsilon Greedy to keep track of best arm so far.
    This model does not consider the context.
    '''

    def __init__(self):
        self.reward = 0.5
        self.num_selected = 1
        self.last_reward = 0


    def update_posterior(self, y):
        # update success/failure counts per observed reward
        if y == 1:
            self.reward += 1
        self.num_selected += 1

    def perform_bernoulli_trials(self, p, n=1):
        """ Perform n Bernoulli trials with success probability p
        and return number of successes."""
        n_success = 0
        for i in range(n):
            trial = random.random()
            if trial < p:
                n_success += 1

        return n_success

    def get_parameters(self, context = None):
        # estimated reward probability for each arm is simply the
        # mean of the current beta distribution
        failure = self.num_selected - self.reward
        est = self.reward / float(self.reward + failure)

        return [self.reward, failure, est]

    def draw_expected_value(self, num_samples = 1):
        return self.reward / float(self.num_selected)


    def save_state(self):
        self.last_reward = self.reward


    def restore_state(self):
        self.reward = self.last_reward


    def reset_state(self):
        self.reward = 0.5
        self.num_selected = 1
        self.last_reward = 0
