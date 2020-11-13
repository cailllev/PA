import src.model.node as n

from typing import Dict, List


class DetectionSystem:
    def __init__(self, name, probs, reset_node, after_nodes):
        # type: (str, Dict[str, float], "n.Node", List[str]) -> None
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
        self._after_nodes = after_nodes

    def get_name(self):
        # type: () -> str
        return self._name

    def get_prob(self):
        # type: () -> float
        return self._p

    def get_init_prob(self):
        # type: () -> float
        return self._p0

    def reset_prob(self):
        # type: () -> None
        self._p = self._p0

    def caught_attacker(self):
        # type: () -> "n.Node"
        """
        updates the chance of catching attacker again next step and returns the node the attacker is set to
        :return: the new pos of the attacker
        """
        self._p *= self._de
        return self._reset_node

    def learn(self):
        # type: () -> None
        """
        0.8 += (1-0.8)/2 => 0.9, ... => 0.95, ...
        """
        self._p += (1.0-self._p) * self._de

    def reset(self):
        # type: () -> None
        self.reset_prob()

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self):
        return self._name \
               + " -> {" + ", ".join(self._after_nodes) + "} !" \
               + " -> " + self._reset_node.get_name()
