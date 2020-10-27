import json
import gym
import random

import src.model.graph as g

from pathlib import Path
path = Path(__file__).parent / "../../config/attack_graphs.json"


def load_rewards(name):
    f = open(path, "r")
    all_data = json.load(f)
    f.close()
    return all_data[name]["rewards"]


# ------------- MODEL ------------- #
graph_name = "simple_webservice"
graph = g.Graph(graph_name, "simple")
nodes = graph.get_nodes()
restartable_nodes = nodes[1:-1]
detection_systems = graph.get_detection_systems()

start_node = nodes[0]
end_node = nodes[-1]

# -------------- GYM -------------- #
rewards = load_rewards(graph_name)


class MTDEnv(gym.Env):
    def __init__(self):
        # ------------- MODEL ------------- #
        self._attacker_pos = start_node

        self._counter = 0
        self._last_time_on_start = 0

        # -------------- GYM -------------- #
        self.progress = []
        self.action_space = gym.spaces.MultiDiscrete([len(restartable_nodes), len(detection_systems)])

    def reset(self):
        self.__init__()

    def step(self, action):
        """
        evaluate the given action, then simulate one time step for the attacker
        action == 0: -> no action, action == 1 -> restart node[0], ..., action == n -> restart node[n-1]

        :param action: the action from the RL agent, what to restart and what to switch
        :return: obs, reward, done, info
        """
        obs = info = None
        reward = 0
        done = False

        self._counter += 1

        # eval action
        restart_node = action[0]
        switch_detection_system = action[1]

        if restart_node:
            node = restartable_nodes[restart_node-1]
            node.get_prev().reset_probs(node)
            reward += rewards["restart_node"]
            if self._attacker_pos == node:
                self._attacker_pos = self._attacker_pos.get_prev()

        if switch_detection_system:
            detection_systems[switch_detection_system-1].reset_prob()
            reward += rewards["switch_detection_system"]

        # simulate next step, "simulated attack"
        val = random.random()
        self._attacker_pos.update_probs()

        # Detection System catching attacker (unless already in honeypot)
        detection_system = self._attacker_pos.get_detection_system()
        if detection_system and not self._attacker_pos.is_honeypot():
            if val < detection_system.get_prob():
                detection_system.caught_attacker()
                self._attacker_pos = start_node

        # Attacker is in honeypot -> Detection System gets better
        if self._attacker_pos.is_honeypot():
            self._attacker_pos.get_detection_system().learn()

        # Attacker getting into next node
        probs = self._attacker_pos.get_probs()
        running_sum = 0
        for node in probs:
            running_sum = running_sum + probs[node]
            if val < running_sum:
                self._attacker_pos.set_compromised(node)
                self._attacker_pos = node
                break

        if self._attacker_pos == start_node:
            self._last_time_on_start = self._counter

        if self._attacker_pos.is_honeypot():
            self._attacker_pos.get_detection_system().learn()

        if self.attacker_wins():
            done = True
            reward += rewards["attacker_wins"]

        # TODO optimize append?
        self.progress.append(self._attacker_pos.get_progress_level())

        return obs, reward, done, info

    def render(self, mode='human'):
        print(self._counter)
        if mode != "human":
            print("Graph:\n" + str(graph))
        print("Attacker: " + self._attacker_pos.get_name())
        print()

    def attacker_wins(self):
        return self._attacker_pos == end_node

    def get_counter(self):
        return self._counter

    def __str__(self):
        return "\n".join([
            str(self._counter),
            "Attacker: " + self._attacker_pos.get_name(),
            "Graph:\n" + str(graph)])


if __name__ == "__main__":
    m = MTDEnv()
    print(str(m))
