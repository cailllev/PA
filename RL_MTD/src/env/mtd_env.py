import json
import gym
import random

import src.model.graph as g


def load_rewards(graph_name):
    f = open("../../config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[graph_name]["rewards"]


# ------------- MODEL ------------- #
graph_name = "simple_webservice"
graph = g.Graph(graph_name, "smiple_attack")
nodes = graph.get_nodes()
detection_systems = graph.get_detection_systems()

start_node = nodes[0]
end_node = nodes[0]

# -------------- GYM -------------- #
rewards = load_rewards(graph_name)

restartable_count = graph.get_restartable_nodes_count()
switchable_count = graph.get_detection_systems_count()


class MTDEnv(gym.Env):
    def __init__(self):
        # ------------- MODEL ------------- #
        self._attacker_pos = start_node

        self._counter = 0
        self._last_time_on_start = 0

        # -------------- GYM -------------- #
        self.action_space = gym.spaces.MultiDiscrete([restartable_count + 1, switchable_count + 1])

    def reset(self):
        self.__init__()

    def step(self, action):
        """
        evaluate the given action, then simulate one timestep for the attacker
        action == 0: -> no action

        :param action: the action from the RL agent, what to restart and what to switch
        :return: obs, reward, done, info
        """
        obs = info = None
        reward = 0
        done = False

        self._counter += 1

        restart_node = action[0]
        switch_detection_system = action[1]

        if restart_node:
            nodes[restart_node-1].reset_probs()
            reward += rewards["restart_node"]

        if switch_detection_system:
            detection_systems[switch_detection_system-1].reset_probs()
            reward += rewards["switch_detection_system"]

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
                self._attacker_pos.set_compromised()
                break

        if self._attacker_pos == start_node:
            self._last_time_on_start = self._counter

        # -------------- GYM -------------- #
        if self.attacker_wins():
            done = True
            reward += rewards["attacker_wins"]

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def update_probs(self):
        self._attacker_pos.update_probs()

    def attacker_wins(self):
        return self._attacker_pos == end_node


if __name__ == "__main__":
    m = MTDEnv()
