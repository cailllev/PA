class Node:
    def __init__(self, name, prev_node, next_nodes):
        self._name = name
        self._prev = prev_node
        self._next = next_nodes
        self._probs = next_nodes

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

    def set_probs(self, probs):
        self._probs = probs

    def get_probs(self):
        return self._probs

    def change_probs(self):
        return self._probs

    def __str__(self):
        return str([
            self._name,
            self._prev.get_name() if self._prev else "null",
            [node.get_name() for node in self._next] if self._next else "null"
            ]) + "\n"
