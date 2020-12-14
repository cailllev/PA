import json
import gym
import random

import src.model.graph as g
import src.model.node as n
import src.model.prevention_system as p

from typing import List, Tuple

from pathlib import Path
path = Path(__file__).parent / "../../config/attack_graphs.json"

# ------------- TO INITIALIZE ------------- #
env_config_init = False

nodes = [n.get_null_node()]
graph = g.get_null_graph()
prevention_systems = p.get_null_prevention_system()

start_node = end_node = n.get_null_node()

rewards = {}
steps_per_simulation = 0
# ----------------------------------------- #


def set_config(graph_name, attack_name):
    # type: (str, str) -> None
    global env_config_init, nodes, graph, prevention_systems, start_node, end_node, rewards, steps_per_simulation

    env_config_init = True

    # ------------- MODEL ------------- #
    graph_name = graph_name
    graph = g.Graph(graph_name, attack_name)
    nodes = graph.get_nodes()
    prevention_systems = graph.get_prevention_systems()

    start_node = nodes[0]
    end_node = nodes[-1]

    # -------------- GYM -------------- #
    rewards, steps_per_simulation = load_gym_config(graph_name)


def load_gym_config(name):
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


def get_prevention_systems_count():
    # type: () -> int
    return len(prevention_systems)


def subtract_one(lst):
    # type: (list) -> list
    for i in range(len(lst)):
        lst[i] = max(lst[i] - 1, 0)


def create_locked_lists(nodes_actions, prevention_system_actions):
    # type: (int, int) -> List[list, list]
    return [[0] * nodes_actions, [0] * prevention_system_actions]


def choose_random_from_list(lst, pause):
    # type: (List[int], int) -> int
    """
    Get a list with ints between 0 and pause. Choose one of the 0's, sets the value to <pause> and returns the index.
    :param lst: a list with ints indicating pauses != 0 or ready == 0
    :param pause: how long a choosen index has to pause
    :return: the index of a random 0 in the list
    """
    valid_action_count = lst.count(0)
    if valid_action_count == 0:
        return 0

    index = random.choice(range(valid_action_count))
    for j in range(len(lst)):
        if index == 0 and lst[j] == 0:

            # found the index'th 0, if it's not the null action, apply pause
            if j != 0:
                lst[j] = pause

            return j

        if lst[j] == 0:
            index -= 1

    raise Exception(f"Invalid list to choose from: {str(lst)}")


class MTDEnv(gym.Env):
    def __init__(self, only_nodes=False, only_prevention_systems=False, nodes_pause=1, prevention_systems_pause=1):
        # type: (bool, bool, int, int) -> None
        """
        own enviroment to simulate behavior of a MTD network
        :param only_nodes: only able to restart nodes, prevention systems are fixed
        :param only_prevention_systems: only able to switch prevention systems, nodes are fixed
        :param nodes_pause: pause between same node restarts (1=every step possible)
        :param prevention_systems_pause: pause between same prevention system switches (1=every step possible)
        """
        if not env_config_init:
            raise Exception("mtd_env module is not configured, call env.set_config() first")

        self._attacker_pos: n.Node = start_node

        self._counter = 0
        self._steps_per_simulation = steps_per_simulation

        self._last_time_on_start = 0
        self._last_action = None
        self._last_reward = 0

        self._null_action_counter = [0, 0]
        self._invalid_action_counter = [0, 0]
        self._total_reward = 0

        # stable baselines
        self._only_nodes = only_nodes
        if only_nodes:
            prevention_systems_actions = 1
        else:
            prevention_systems_actions = get_prevention_systems_count() + 1

        self._only_prevention_systems = only_prevention_systems
        if only_prevention_systems:
            nodes_actions = 1
        else:
            nodes_actions = get_restartable_nodes_count() + 1

        self.action_space = gym.spaces.MultiDiscrete([nodes_actions, prevention_systems_actions])
        self.observation_space = gym.spaces.Discrete(graph.get_obs_range())

        self._locked_nodes, self._locked_prevention_systems = create_locked_lists(nodes_actions,
                                                                                 prevention_systems_actions)
        self._nodes_pause = nodes_pause
        self._prevention_systems_pause = prevention_systems_pause

    # ------------------------- GYM ------------------------- #
    def reset(self):
        # type: () -> int
        graph.reset()
        self.__init__(self._only_nodes, self._only_prevention_systems, self._nodes_pause, self._prevention_systems_pause)
        return 0

    def step(self, action):
        # type: (List[int, int]) -> (int, int, bool, dict)
        """
        evaluate the given action, then simulate one time step for the attacker
        action in Discrete: 0...restartable_nodes+prevention_systems --> parse to MultiDiscrete
        action in MultiDiscrete: [0...restartable_nodes, 0...prevention_systems]

        action[0] == 0: -> no action, action[0] == 1 -> restart node[1], ..., action[0] == n -> restart node[n]
        action[1] == 0: -> no action, action[1] == 1 -> restart prevention system[0], ...

        :param action: the action from the RL agent, what to restart and what to switch
        :return: obs, reward, done, info
        """
        obs = 0  # index of attacker pos before step simulation (0=unknown)
        reward = 0
        done = False

        self._counter += 1

        subtract_one(self._locked_nodes)
        subtract_one(self._locked_prevention_systems)

        # --------------- parse action --------------- #
        restart_node = action[0]
        switch_prevention_system = action[1]

        # check if action is invalid
        invalid_action = [0, 0]
        if self._only_prevention_systems and restart_node:
            invalid_action[0] = 1
        elif self._locked_nodes[restart_node] > 0:
            invalid_action[0] = 1

        if self._only_nodes and switch_prevention_system:
            invalid_action[1] = 1
        elif self._locked_prevention_systems[switch_prevention_system] > 0:
            invalid_action[1] = 1

        # https://github.com/hill-a/stable-baselines/issues/108
        # TL;DR: if there is a invalid action, penalize it and do nothing
        # only penalize in learning, not in simulating
        if invalid_action[0]:
            reward += rewards["invalid_action"]
            self._invalid_action_counter[0] += 1
            restart_node = 0

        if invalid_action[1]:
            reward += rewards["invalid_action"]
            self._invalid_action_counter[1] += 1
            switch_prevention_system = 0

        self._last_action = [restart_node, switch_prevention_system]

        # --------------- eval action --------------- #
        if restart_node:
            self._locked_nodes[restart_node] = self._nodes_pause

            node = nodes[restart_node]
            node.get_prev().reset_probs(node)
            reward += rewards["restart_node"]
            if self._attacker_pos == node:
                self._attacker_pos = self._attacker_pos.get_prev()
        else:
            self._null_action_counter[0] += 1

        if switch_prevention_system:
            self._locked_prevention_systems[switch_prevention_system] = self._prevention_systems_pause

            prevention_systems[switch_prevention_system-1].reset_prob()
            reward += rewards["switch_prevention_system"]
        else:
            self._null_action_counter[1] += 1

        # ---------------- simulate ---------------- #
        self._attacker_pos.update_probs()

        # Prevention System catching attacker (unless already in honeypot)
        caught = False
        prevention_system = self._attacker_pos.get_prevention_system()
        if prevention_system and not self._attacker_pos.is_honeypot():
            if prevention_system.get_prob() > random.random():
                obs = self._attacker_pos.get_index()
                self._attacker_pos = prevention_system.caught_attacker()
                caught = True

        # Attacker is in honeypot -> Prevention System gets better
        if self._attacker_pos.is_honeypot():
            self._attacker_pos.get_prevention_system().learn()
            obs = self._attacker_pos.get_index()

        # Attacker getting into next node, only possible if not caught
        if not caught:
            probs = self._attacker_pos.get_probs()
            biggest_prob = 0
            biggest_prob_node = None
            for next_node, prob in probs.items():
                val = random.random()
                # attacker getting into next
                if prob > val:
                    # which next is most likely?
                    surplus = (prob - val) / prob
                    if surplus > biggest_prob:
                        biggest_prob = surplus
                        biggest_prob_node = next_node

            if biggest_prob_node:
                self._attacker_pos.set_compromised(biggest_prob_node)
                self._attacker_pos = biggest_prob_node

        # ---------------- finalize ---------------- #
        reward += rewards["bias_per_step"]

        if not self._attacker_pos.is_honeypot():
            reward += self._attacker_pos.get_progress_level() * rewards["progression"]

        if self._attacker_pos == start_node:
            self._last_time_on_start = self._counter

        if self.attacker_wins():
            done = True
            reward += rewards["attacker_wins"]

        if self.defender_wins():
            done = True

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
            switched = prevention_systems[self._last_action[1]].get_name()
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

    def get_counter(self):
        # type: () -> int
        return self._counter

    def get_steps_per_simulation(self):
        # type: () -> int
        return self._steps_per_simulation

    def get_total_reward(self):
        # type: () -> int
        return self._total_reward

    def get_last_action(self):
        # type: () -> (int, int)
        return self._last_action

    def get_null_actions_count(self):
        # type: () -> (int, int)
        return self._null_action_counter

    def get_invalid_actions_count(self):
        # type: () -> (int, int)
        return self._invalid_action_counter

    def get_invalid_actions_penalty(self):
        # type: () -> int
        return sum(self._invalid_action_counter) * rewards["invalid_action"]

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
    set_config("simple_webservice", "professional")
    m = MTDEnv()
    print(str(m))

    m.step([0, 0])
    print(str(m))
