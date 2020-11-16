import random
from typing import List

from src.env.mtd_env import get_restartable_nodes_count, get_detection_systems_count, nodes, detection_systems, \
    subtract_one, create_locked_lists


def get_detection_system_index_from_obs(obs):
    """
    gets an observation and determines which detection system caugth the attacker
    :param obs: the index of the node the attacker was caught on
    :return: the index of the corresponding detection system
    """
    detection_system = nodes[obs]

    for i in range(len(detection_systems)):
        if detection_system == detection_systems[i]:
            return i

    raise Exception(f"No detection system found for node {nodes[obs].get_name()}.")


class Defender2000:
    # TODO verify logic
    def __init__(self, only_nodes=False, only_detection_systems=False, nodes_pause=1, detection_systems_pause=1):
        # type: (bool, bool, int, int) -> None
        """
        Own implementation of an agent for given mtd problem. Actions are taken in 2 scenarios:
        1. Detection system catches an attacker -> switch IDS, restart nodes up to incident point, one restart per step
            a) Only does restarts and switched if valid (i.e. not in pause)
            b) If not all actions done and new and deeper incident point detected, remove all old actions and create new
                 actions for the new incident point
        2. Randomly restart a node or switch a detection system
        :param only_nodes: only nodes are restartable
        :param only_detection_systems: only detection systems are switchable
        :param nodes_pause: how long between restarts of the same node (1 = every step possible)
        :param detection_systems_pause: how long between switches of the same detection system (1 = every step possible)
        """
        self._next_actions = []

        self._random_restart_threshold = 0.4
        self._random_switch_threshold = 0.2

        if not only_nodes:
            self._nodes_count = get_restartable_nodes_count()
        else:
            self._nodes_count = 0

        if not only_detection_systems:
            self._detection_systems_count = get_detection_systems_count()
        else:
            self._detection_systems_count = 0

        self._nodes_pause = nodes_pause
        self._detection_systems_pause = detection_systems_pause

        self._locked_nodes, self._locked_detection_systems = create_locked_lists()

    def _set_next_actions(self, obs):
        # type: (int) -> None
        """
        sets the next actions, if attacker e.g. is caught at progress lvl 3 (and kicked out) do the following:
        1. switch the Intrusion Detection System (IDS) so the same attack will most likely fail and the current node
        2. restart the nodes before current node, -> 2, 1
        actions => [[3,1],[2,0],[1,0]]
        :param obs: the compromised node the attacker was caught on
        """
        ds_index = get_detection_system_index_from_obs(obs)

        self._next_actions = []
        # add all nodes from incident point
        for i in range(obs, 0, -1):
            if not nodes[i].is_honeypot():
                self._next_actions.append([i, 0])

        if len(self._next_actions) == 0:
            self._next_actions = [[0, 0]]

        self._next_actions[0][1] = ds_index

    def _get_next_action(self):
        # type: () -> List[int]
        """
        if still actions to perform, return the next to perform, else return random
        :return:
        """
        if self._next_actions:
            action = self._next_actions[0]
            return self._check_not_locked(action)

        else:
            val = random.random()
            action = [0, 0]

            if val > self._random_restart_threshold:
                action[0] = random.choice(range(self._nodes_count)) + 1

            if val > self._random_switch_threshold:
                action[1] = random.choice(range(self._detection_systems_count)) + 1

            return self._check_not_locked(action)

    def _check_not_locked(self, action):
        # type: (List[int]) -> List[int]
        """
        Gets an action, checks if the action does not violate the pause. If the action is ok, set the pause, if action
        is not okay, return 0
        :param action: what to do
        :return: the action if possible or 0 if not
        """
        if action[0] > 0 and self._locked_nodes[action[0]] == 0:
            self._locked_nodes[action[0]] = self._nodes_pause
            self._shift_left(0)
        else:
            action[0] = 0

        if action[1] > 0 and self._locked_detection_systems[action[1]] == 0:
            self._locked_detection_systems[action[1]] = self._detection_systems_pause
            self._shift_left(1)
        else:
            action[1] = 0

        return action

    def _shift_left(self, action_type):
        # type: (int) -> None
        """
        action_type=0: [[2,1], [1,0]] --> [[1,1], [0,0]]
        action_type=1: [[2,1], [1,0]] --> [[2,0], [1,0]]
        """
        # copy next to current
        for i in range(len(self._next_actions)-2):
            self._next_actions[i][action_type] = self._next_actions[i + 1][action_type]

        # set last to null
        if self._next_actions:
            self._next_actions[-1][action_type] = 0

    def predict(self, obs):
        # type: (int) -> List[int]
        """
        set next actions when obs > 0 --> attacker was caught, else pick random
        when attacker is caught from internet to first node, don't switch
        :param obs: where the attacker is (or 0 is unknown)
        :return: an action
        """
        # removes one "pause step" in the locked nodes and detection systems, i.e. locked_nodes: [3,1,1,0] -> [2,0,0,0]
        subtract_one(self._locked_nodes)
        subtract_one(self._locked_detection_systems)

        # if an attacker is caught deeper in the system, do restart from there
        if obs > 0:
            if self._next_actions and obs > self._next_actions[0][0]:
                self._set_next_actions(obs)

        return self._get_next_action()
