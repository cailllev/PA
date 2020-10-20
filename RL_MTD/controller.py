from model.graph import Graph

import math


class Controller:
    def __init__(self):
        self._graph = Graph("simple_webservice")
        self._nodes = self._graph.get_nodes()

        self._start_node = self._nodes[0]
        self._end_node = self._nodes[0]
        self._attacker_pos = self._start_node

    def iterate(self):
        pass

    def get_current_probs(self):
        self._attacker_pos.get_probs()

    def change_probs(self):
        self._graph.change_probs(self._attacker_pos)

    def has_attacker_won(self):
        return self._attacker_pos == self._end_node

