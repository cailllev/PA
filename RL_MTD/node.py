class Node:
    def __init__(self, name, prev_node):
        self._name = name
        self._prev = prev_node
        self._next = []
        self._dt = {}  # {next_node: dt, next_node: dt}

    def get_name(self):
        return self._name

    def get_prev(self):
        return self._prev

    def get_next(self):
        return self._next

    def set_next(self, next_node):
        self._next.append(next_node)
