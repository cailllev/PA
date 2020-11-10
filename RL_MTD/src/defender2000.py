import random

from typing import List


class Defender2000:
    def __init__(self, nodes_count, detection_systems_count):
        # type: (int, int) -> None
        self._next_actions = []

        self._random_restart_threshold = 0.95
        self._random_switch_threshold = 0.8

        self._nodes_count = nodes_count
        self._detection_systems_count = detection_systems_count

    def _clear_next_actions(self):
        # type: () -> None
        self._next_actions = []

    def _set_next_actions(self, obs):
        # type: (int) -> None
        """
        sets the next actions, if attacker e.g. is caught at progress lvl 3 (and kicked out) do the following to assure
        security:
        1. switch the Intrusion Detection System (IDS) so the same attack will most likely fail
        2. restart the node the attacker was caught on (to minimize his chance to hack it again), then restart the node
            before current node untill the first node in the service is reached, i.e. 3, 2, 1
        :param obs: the compromised node the attacker was caught on
        :return:
        """
        restart_nodes = []
        for i in range(obs, 0, -1):
            restart_nodes.append([i, 0])
        self._next_actions = [[0, 1], *restart_nodes]  # [switch IDS, restart nodes]

    def _get_next_action(self):
        # type: () -> List[int]
        """
        if still actions to perform, return the next to perform, else return random
        :return:
        """
        if self._next_actions:
            return self._next_actions.pop(0)

        else:
            val = random.random()
            action = [0, 0]

            if val > self._random_restart_threshold:
                action[0] = random.choice(range(self._nodes_count)) + 1

            if val > self._random_switch_threshold:
                action[1] = random.choice(range(self._detection_systems_count)) + 1

            return action

    def predict(self, obs):
        # type: (int) -> List[int]
        """
        set next actions when obs > 0 --> attacker was caught, else pick random
        :param obs:
        :return:
        """
        if obs > 0:
            self._clear_next_actions()
            self._set_next_actions(obs)

        return self._get_next_action()
