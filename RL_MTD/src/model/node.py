import src.model.detection_system as d

from typing import Union
from typing import Dict


class Node:
    def __init__(self, name, prev_node, next_nodes, progress, is_honeypot):
        # type: (str, Union["Node", None], dict, int, bool) -> None
        """
        :param name: name of the node
        :param prev_node: the previous node (in case of this.reset)
        :param next_nodes: the next nodes (and the prob to get to them)
        :param progress: the relative progress of the attack (when the attacker is at this node)
        :param is_honeypot: if this node is a honeypot
        """
        self._name = name
        self._prev = prev_node
        self._next = next_nodes
        self._detection_system = None
        self._progress = progress
        self._is_honeypot = is_honeypot

    def get_name(self):
        # type: () -> str
        return self._name

    def set_prev(self, node):
        # type: ("Node") -> None
        self._prev = node

    def get_prev(self):
        # type: () -> "Node"
        return self._prev

    def set_next(self, next_nodes):
        # type: (Dict["Node", Dict[str, float]]) -> None
        self._next = next_nodes

    def get_next(self):
        # type: () -> dict
        return self._next

    def get_probs(self):
        # type: () -> dict
        """
        gets the next nodes and the probability to get into the next nodes
        :return: {planner: 0.54, authorizer_honeypot: 0.21}
        """
        return {node: self._next[node]["current"] for node in self._next}

    def reset_probs(self, next_node):
        # type: ("Node") -> None
        """
        sets the probability to reach the next_node from the current node to init
        """
        self._next[next_node]["current"] = self._next[next_node]["init"]

    def update_probs(self):
        # type: () -> None
        """
        updates the current probability to get into the next node, current += dt
        """
        for node in self._next:
            self._next[node]["current"] += self._next[node]["dt"]

    def set_compromised(self, next_node):
        # type: ("Node") -> None
        """
        sets the chance to get to next node to 1 (next node is compromised)
        """
        self._next[next_node]["current"] = 1

    def set_detection_system(self, detection_system):
        # type: (d.DetectionSystem) -> None
        self._detection_system = detection_system

    def get_detection_system(self):
        # type: () -> d.DetectionSystem
        return self._detection_system

    def get_progress_level(self):
        # type: () -> int
        return self._progress

    def is_honeypot(self):
        # type: () -> bool
        return self._is_honeypot

    def reset(self):
        # type: () -> None
        """
        resets all probs outgoing from this node
        """
        for next_node in self._next:
            self.reset_probs(next_node)

    def __str__(self):
        """
        :return: name of current, name of previous node, name of next node(s),
        """
        return " -> ".join([
            self._prev.get_name() if self._prev else "null",
            "*" + self._name + "*",
            "{" + ", ".join([node.get_name() for node in self._next]) + "}" if self._next else "{null}"
            ])
