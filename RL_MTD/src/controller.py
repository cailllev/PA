import src.model.graph as g

import random


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
            self.services()
            self.mtd_actions()
            if self.attacker_wins():
                return False

            self.update_probs()

        return True

    def iterate(self):
        probs = self._attacker_pos.get_probs()

        chance = random.random()
        if chance <= probs():
            pass

    def services(self):
        pass

    def mtd_actions(self):
        switched = []

        for node in switched:
            node.reset_probs()

    def update_probs(self):
        self._attacker_pos.update_probs()

    def attacker_wins(self):
        return self._attacker_pos == self._end_node


if __name__ == "__main__":
    c = Controller()
    c.start()
