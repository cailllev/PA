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
        val = random.random()

        # probs = {planner: 0.6, authorizer_honeypot: 0.2}
        # split = 0, 0.6, 0.8
        # (val < split) -> change to node
        # val: 0.7 -> node: authorizer_honeypot

        probs = self._attacker_pos.get_probs()
        running_sum = 0
        for node in probs:
            running_sum = running_sum + probs[node]
            if val < running_sum:
                self._attacker_pos = node
                break

    def services(self):
        pass

    def mtd_actions(self):
        switched = []

        for node in switched:
            node.reset_probs()
            if self._attacker_pos == node:
                # TODO
                pass  # reset attacker to previous node

    def update_probs(self):
        self._attacker_pos.update_probs()

    def attacker_wins(self):
        return self._attacker_pos == self._end_node


if __name__ == "__main__":
    c = Controller()
    c.start()
