class Node:
    def __init__(self, name, prev_node, next_nodes):
        self._name = name
        self._prev = prev_node
        self._next = next_nodes

    def get_name(self):
        return self._name

    def set_prev(self, node):
        self._prev = node

    def get_prev(self):
        return self._prev

    def set_next(self, next_nodes):
        self._next = next_nodes

    def get_next(self):
        return self._next

    def reset_probs(self):
        """
        sets the current prob to init prob
        """
        for node in self._next:
            self._next[node]["current"] = self._next[node]["init"]

    def update_probs(self):
        """
        updates the current probability to get into the next node, current += dt
        """
        for node in self._next:
            self._next[node]["current"] = self._next[node]["current"] + self._next[node]["dt"]

    def get_probs(self):
        """
        gets the next nodes and the probability to get into the next nodes
        :return: {planner: 0.54, authorizer_honeypot: 0.21}
        """
        return {node: self._next[node]["current"] for node in self._next}

    def __str__(self):
        """
        :return: name of current, name of previous node, name of next node(s)
        """
        return str([
            self._name,
            self._prev.get_name() if self._prev else "null",
            [node.get_name() for node in self._next] if self._next else "null"
            ]) + "\n"
