class DetectionSystem:
    def __init__(self, name, probs, reset_node):
        self._name = name
        self._p0 = probs["init"]
        self._p = probs["init"]
        self._reset_node = reset_node

    def get_name(self):
        return self._name

    def reset_prob(self):
        self._p = self._p0

    def caught_attacker(self):
        self._p /= 2.0

    def learn(self):
        """
        0.8 += (1-0.8)/2 => 0.9, ... => 0.95, ...
        """
        self._p += (1.0-self._p)/2.0

    def get_prob(self):
        return self._p

    def get_reset_node(self):
        return self._reset_node

    def __str__(self):
        return self._name + " -> " + self._reset_node.get_name()
