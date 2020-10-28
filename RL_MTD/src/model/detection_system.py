import src.model.node as n

from typing import Dict


class DetectionSystem:
    def __init__(self, name, probs, reset_node):
        # type: (str, Dict[str, float], "n.Node") -> None
        """
        :param name: the name of this system
        :param probs: the prob of catching attacker and its change after catching attacker
        :param reset_node: where the attacker is after they're caught
        """
        self._name = name
        self._p0 = probs["init"]
        self._p = probs["init"]
        self._de = probs["de"]
        self._reset_node = reset_node

    def get_name(self):
        # type: () -> str
        return self._name

    def reset_prob(self):
        # type: () -> None
        self._p = self._p0

    def caught_attacker(self):
        # type: () -> None
        self._p *= self._de

    def learn(self):
        # type: () -> None
        """
        0.8 += (1-0.8)/2 => 0.9, ... => 0.95, ...
        """
        self._p += (1.0-self._p) * self._de

    def get_prob(self):
        # type: () -> float
        return self._p

    def get_reset_node(self):
        # type: () -> n.Node
        return self._reset_node

    def __str__(self):
        return self._name + " -> " + self._reset_node.get_name()
