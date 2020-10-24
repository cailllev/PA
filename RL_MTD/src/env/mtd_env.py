import src.model.graph as g
import random

import gym
from gym import spaces


class MTDEnv(gym.Env):
    def __init__(self):
        # ------------- MODEL ------------- #
        self._graph = g.Graph("simple_webservice", "smiple_attack")
        self._nodes = self._graph.get_nodes()

        self._start_node = self._nodes[0]
        self._end_node = self._nodes[0]
        self._attacker_pos = self._start_node

        self._counter = 0
        self._last_time_on_start = 0

        self._restarted_nodes = []

        # -------------- GYM -------------- #
        restartable_count = self._graph.get_restartable_nodes_count()
        switchable_count = self._graph.get_detection_systems_count()
        self.action_space = spaces.MultiDiscrete([restartable_count, switchable_count])

    def reset(self):
        self.__init__()

    def step(self, action):
        """
        evaluate the given action, then simulate one timestep for the attacker
        :param action: the action from the RL agent, what to restart and what to switch
        :return: obs, reward, done, info
        """
        # TODO eval action
        self._counter = self._counter + 1

        # simulate next step
        val = random.random()

        # probs = {planner: 0.6, authorizer_honeypot: 0.2}
        # split = 0, 0.6, 0.8
        # (val < split) -> change to node
        # val: 0.7 -> node: authorizer_honeypot

        probs = self._attacker_pos.get_probs()
        running_sum = 0
        for node in probs:
            running_sum = running_sum + probs[node]
            if val < running_sum:
                self._attacker_pos = node
                break

    def render(self, mode='human'):
        pass

    def update_probs(self):
        self._attacker_pos.update_probs()

    def attacker_wins(self):
        return self._attacker_pos == self._end_node


if __name__ == "__main__":
    m = MTDEnv()
