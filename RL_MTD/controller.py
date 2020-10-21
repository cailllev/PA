import model.graph as g

import random


def change_probs(node):
    node.change_probs()


class Controller:
    def __init__(self):
        self._graph = g.Graph("simple_webservice", "smiple_attack")
        self._nodes = self._graph.get_nodes()

        self._start_node = self._nodes[0]
        self._end_node = self._nodes[0]
        self._attacker_pos = self._start_node

    def start(self):
        count = 0
        while count < 1000:
            self.iterate()
            if self.attacker_won():
                return False

        return True

    def iterate(self):
        probs = self._attacker_pos.get_probs()

        chance = random.random()
        if chance <= self.get_current_probs():
            pass

    def get_current_probs(self):
        return self._attacker_pos.get_probs()

    def attacker_won(self):
        return self._attacker_pos == self._end_node


if __name__ == "__main__":
    c = Controller()
    c.start()
