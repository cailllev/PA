import json
import gym
import random
from collections import deque

import src.model.graph as g

from typing import List, Tuple

from pathlib import Path
path = Path(__file__).parent / "../../config/attack_graphs.json"


def load_config(name):
    # type: (str) -> Tuple[dict, int]
    f = open(path, "r")
    all_data = json.load(f)
    f.close()
    return all_data[name]["rewards"], all_data[name]["steps_per_simulation"]


def get_restartable_nodes_count():
    # type: () -> int
    """
    first and last node are never restartable (start and finish)
    """
    return len(nodes) - 2


def get_detection_systems_count():
    # type: () -> int
    return len(detection_systems)


# ------------- MODEL ------------- #
graph_name = "simple_webservice"
graph = g.Graph(graph_name, "simple")
nodes = graph.get_nodes()
detection_systems = graph.get_detection_systems()

start_node = nodes[0]
end_node = nodes[-1]

# -------------- GYM -------------- #
rewards, steps_per_simulation = load_config(graph_name)
bias_per_step = rewards["bias_per_step"]


class MTDEnv(gym.Env):
    def __init__(self):
        # type: () -> None
        self._attacker_pos = start_node

        self._counter = 0
        self._steps_per_simulation = steps_per_simulation

        self._last_time_on_start = 0
        self._last_action = None
        self._last_reward = 0

        self._progress_history = deque()
        self._null_action_counter = 0
        self._total_reward = 0

        # stable baselines
        self.action_space = gym.spaces.MultiDiscrete([get_restartable_nodes_count() + 1,
                                                      get_detection_systems_count() + 1])
        self.observation_space = gym.spaces.Discrete(graph.get_progress_levels_count())

    # ------------------------- GYM ------------------------- #
    def reset(self):
        # type: () -> int
        self.__init__()
        return 0

    def step(self, action):
        # type: (List[int]) -> Tuple[int, int, bool, dict]
        """
        evaluate the given action, then simulate one time step for the attacker
        action in Discrete: 0...restartable_nodes+detection_systems --> parse to MultiDiscrete
        action in MultiDiscrete: [0...restartable_nodes, 0...detection_systems]

        action[0] == 0: -> no action, action[0] == 1 -> restart node[1], ..., action[0] == n -> restart node[n]
        action[1] == 0: -> no action, action[1] == 1 -> restart detection system[0], ...

        :param action: the action from the RL agent, what to restart and what to switch
        :return: obs, reward, done, info
        """
        obs = 0  # progress of attacker before step simulation (0=unknown)
        reward = bias_per_step
        done = False

        self._counter += 1
        self._last_action = action

        # --------------- eval action --------------- #
        restart_node = action[0]
        switch_detection_system = action[1]

        if restart_node:
            node = nodes[restart_node]
            node.get_prev().reset_probs(node)
            reward += rewards["restart_node"]
            if self._attacker_pos == node:
                self._attacker_pos = self._attacker_pos.get_prev()

        if switch_detection_system:
            detection_systems[switch_detection_system-1].reset_prob()
            reward += rewards["switch_detection_system"]

        # ---------------- simulate ---------------- #
        val = random.random()
        self._attacker_pos.update_probs()

        # Detection System catching attacker (unless already in honeypot)
        caught = False
        detection_system = self._attacker_pos.get_detection_system()
        if detection_system and not self._attacker_pos.is_honeypot():
            if val < detection_system.get_prob():
                detection_system.caught_attacker()
                obs = self._attacker_pos.get_progress_level()
                self._attacker_pos = start_node
                caught = True
                reward += rewards["caught_attacker"]

        # Attacker is in honeypot -> Detection System gets better
        # & remove progression penalty from reward (gets added back later)
        if self._attacker_pos.is_honeypot():
            self._attacker_pos.get_detection_system().learn()
            obs = self._attacker_pos.get_progress_level()
            reward -= self._attacker_pos.get_progress_level() * rewards["progression"]

        # Attacker getting into next node, only possible if not caught
        if not caught:
            probs = self._attacker_pos.get_probs()
            running_sum = 0
            for node in probs:
                running_sum = running_sum + probs[node]
                if val < running_sum:
                    self._attacker_pos.set_compromised(node)
                    self._attacker_pos = node
                    break

        # ---------------- finalize ---------------- #
        reward += self._attacker_pos.get_progress_level() * rewards["progression"]

        if self._attacker_pos == start_node:
            self._last_time_on_start = self._counter

        if self.attacker_wins():
            done = True
            reward += rewards["attacker_wins"]

        if self.defender_wins():
            done = True
            reward += rewards["defender_wins"]

        self._progress_history.append(self._attacker_pos.get_progress_level())

        if action is [0, 0]:
            self._null_action_counter += 1

        self._total_reward += reward
        self._last_reward = reward

        info = {
            "attacker_pos": self._attacker_pos.get_name(),
            "counter": self._counter,
            "last_time_on_start": self._last_time_on_start
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        # type: (str) -> None
        if self._last_action[0]:
            restarted = nodes[self._last_action[0]].get_name()
        else:
            restarted = "-"

        if self._last_action[1]:
            switched = detection_systems[self._last_action[1]].get_name()
        else:
            switched = "-"

        s = f"{self._counter},".ljust(6)
        s += f"reward: {self._last_reward},".ljust(15)
        s += f"restarted: {restarted},".ljust(35)
        s += f"switched: {switched},".ljust(20)
        s += f"attacker at: {self._attacker_pos.get_name()}".ljust(40)

        print(s)

    # ------------------------- OWN ------------------------- #
    def attacker_wins(self):
        # type: () -> bool
        return self._attacker_pos == end_node

    def defender_wins(self):
        # type: () -> bool
        return self._counter >= self._steps_per_simulation and not self.attacker_wins()

    def get_progress_history(self):
        # type: () -> List[int]
        return list(self._progress_history)

    def get_counter(self):
        # type: () -> int
        return self._counter

    def get_steps_per_simulation(self):
        # type: () -> int
        return self._steps_per_simulation

    def get_total_reward(self):
        # type: () -> int
        return self._total_reward

    def get_null_action_ratio(self):
        # type: () -> float
        return self._null_action_counter / self._counter

    def __str__(self):
        # type: () -> str
        if self.attacker_wins():
            winner = "attacker"
        elif self.defender_wins():
            winner = "defender"
        else:
            winner = "still running..."

        return f"\nwinner:        {winner}\n" \
               f"steps:         {self._counter}\n" \
               f"total_reward:  {self._total_reward}"


if __name__ == "__main__":
    m = MTDEnv()
    print(str(m))
