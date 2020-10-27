class DetectionSystem:
    def __init__(self, name, probs, reset_node):
        self._name = name
        self._p0 = probs["init"]
        self._p = probs["init"]
        self._dt = probs["dt"]
        self._reset_node = reset_node

    def get_name(self):
        return self._name

    def reset_prob(self):
        self._p = self._p0

    def update_prob(self):
        self._p += self._dt

    def get_prob(self):
        return self._p

    def get_all_probs(self):
        return [self._p0, self._p, self._dt]

    def get_reset_node(self):
        return self._reset_node

    def __str__(self):
        return ", ".join([self._name, str(self._reset_node)])
