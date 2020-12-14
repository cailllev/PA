from copy import copy

import src.env.mtd_env as env_constants


def get_detection_system_index_from_obs(obs):
    """
    gets an observation and determines which detection system caugth the attacker
    :param obs: the index of the node the attacker was caught on
    :return: the index of the corresponding detection system
    """
    detection_system = env_constants.nodes[obs].get_detection_system()

    for i in range(len(env_constants.detection_systems)):
        if detection_system == env_constants.detection_systems[i]:
            return i

    raise Exception(f"No detection system found for node {env_constants.nodes[obs].get_name()}.")


class Defender2000:
    def __init__(self, only_nodes=False, only_detection_systems=False, nodes_pause=1, detection_systems_pause=1):
        # type: (bool, bool, int, int) -> None
        """
        Own implementation of an agent for given mtd problem. Actions are taken in 2 scenarios:
        1. Detection system catches an attacker -> switch IPS, restart nodes up to incident point, one restart per step
            a) Only does planned restarts and switched if valid (i.e. not in pause), else do a random valid one
            b) If not all actions done but new and deeper incident point detected, remove all old actions and create new
                 actions for the new incident point
        2. Randomly restart a node and / or switch a detection system
        :param only_nodes: only nodes are restartable
        :param only_detection_systems: only detection systems are switchable
        :param nodes_pause: how long between restarts of the same node (1 = every step possible)
        :param detection_systems_pause: how long between switches of the same detection system (1 = every step possible)
        """
        self._next_actions = []

        if not only_nodes:
            self._detection_systems_count = env_constants.get_detection_systems_count()
        else:
            self._detection_systems_count = 0

        if not only_detection_systems:
            self._nodes_count = env_constants.get_restartable_nodes_count()
        else:
            self._nodes_count = 0

        self._nodes_pause = nodes_pause
        self._detection_systems_pause = detection_systems_pause

        self._locked_nodes, self._locked_detection_systems = \
            env_constants.create_locked_lists(self._nodes_count + 1, self._detection_systems_count + 1)

    def _set_next_actions(self, obs):
        # type: (int) -> None
        """
        sets the next actions, if attacker e.g. is caught at progress lvl 3 (and kicked out) do the following:
        1. switch the Intrusion Detection System (IPS) so the same attack will most likely fail and the current node
        2. restart the nodes before current node, -> 2, 1
        actions => [[3,1],[2,0],[1,0]]
        :param obs: the compromised node the attacker was caught on
        """

        self._next_actions = []
        if self._nodes_count:
            # add all nodes from incident point
            for i in range(obs, 0, -1):
                if not env_constants.nodes[i].is_honeypot():
                    self._next_actions.append([i, 0])

        if len(self._next_actions) == 0:
            self._next_actions = [[0, 0]]

        if self._detection_systems_count:
            ds_index = get_detection_system_index_from_obs(obs)
            self._next_actions[0][1] = ds_index + 1

    def _get_next_action(self):
        # type: () -> (int, int)
        """
        if still actions to perform, return the next to perform, else return random
        :return:
        """
        if self._next_actions:
            # Checks if the next action does not violate the pause.
            # If the action is ok, set it, shift all next actions left (for next step) and set new pause.
            # If it's not ok, choose another random action and check for original in next step.
            # Actions get overwritten if deeper incident is detected.
            action = copy(self._next_actions[0])

            if action[0] > 0 and self._locked_nodes[action[0]] == 0:
                self._locked_nodes[action[0]] = self._nodes_pause
                self._shift_left(0)
            else:
                action[0] = env_constants.choose_random_from_list(self._locked_nodes, self._nodes_pause)

            if action[1] > 0 and self._locked_detection_systems[action[1]] == 0:
                self._locked_detection_systems[action[1]] = self._detection_systems_pause
                self._shift_left(1)
            else:
                action[1] = env_constants.choose_random_from_list(self._locked_detection_systems,
                                                                  self._detection_systems_pause)

            return action

        else:
            action = [0, 0]

            # do only try to get action if there are restarable nodes
            if self._nodes_count:
                action[0] = env_constants.choose_random_from_list(self._locked_nodes, self._nodes_pause)

            # same with detection systems
            if self._detection_systems_count:
                action[1] = env_constants.choose_random_from_list(self._locked_detection_systems,
                                                                  self._detection_systems_pause)

            return action

    def _shift_left(self, action_type):
        # type: (int) -> None
        """
        action_type=0: [[2,1], [1,0]] --> [[1,1], [0,0]]
        action_type=1: [[2,1], [1,0]] --> [[2,0], [1,0]]
        """
        # copy next to current
        for i in range(len(self._next_actions)-1):
            self._next_actions[i][action_type] = self._next_actions[i + 1][action_type]

        # set last to null
        if self._next_actions:
            self._next_actions[-1][action_type] = 0

            if self._next_actions[-1] == [0, 0]:
                self._next_actions = self._next_actions[0:-1]

    def predict(self, obs):
        # type: (int) -> (int, int)
        """
        set next actions when obs > 0 --> attacker was caught, else pick random
        when attacker is caught from internet to first node, don't switch
        :param obs: where the attacker is (or 0 is unknown)
        :return: an action
        """
        # removes one "pause step" in the locked nodes and detection systems, i.e. locked_nodes: [3,1,1,0] -> [2,0,0,0]
        env_constants.subtract_one(self._locked_nodes)
        env_constants.subtract_one(self._locked_detection_systems)

        if obs > 0:
            if self._next_actions:
                # when an attacker is caught deeper in the system, do restart from there, else just do the saved actions
                if obs > self._next_actions[0][0]:
                    self._set_next_actions(obs)

            # no next actions registered but there is an obs
            else:
                self._set_next_actions(obs)

        return self._get_next_action(), None
