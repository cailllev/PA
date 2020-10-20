import json
from model.node import *


def load_json(name_graph):
    f = open("../config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[name_graph]


def load_nodes(nodes_data):
    nodes = []
    for key in nodes_data:
        n = NodeData(key, nodes_data[key])
        nodes.append(n)
    return nodes


def load_attack(attack_data):
    pass


class NodeData:
    def __init__(self, name, data):
        self.name = name
        self.prev = data["previous"]
        self.next = data["next"]


def init_graph(nodes_data):
    nodes = []

    # init nodes and previous nodes
    for node in nodes_data:
        n = Node(node.name, node.prev, node.next)
        nodes.append(n)

    # set reference to previous node
    for node in nodes:
        prev_name = node.get_prev()
        for pot_node in nodes:
            if pot_node.get_name() == prev_name:
                node.set_prev(pot_node)

    # set reference and probability to next nodes
    for node in nodes:
        next_nodes = []
        probs = {}

        next_names = node.get_next()
        if next_names:

            for next_name in next_names:
                for pot_node in nodes:
                    if pot_node.get_name() == next_name:
                        next_nodes.append(pot_node)
                        probs[next_name] = next_names[next_name]

            node.set_next(next_nodes)
            node.set_probs(probs)

    return nodes


class Graph:
    def __init__(self, name_graph, name_attack):
        graph_data = load_json(name_graph)

        nodes_data = load_nodes(graph_data["nodes"])
        self._nodes = init_graph(nodes_data)
        self.init_probs(name_attack)

    def init_probs(self, name_attack):
        for node in nodes:
            pass

    def get_nodes(self):
        return self._nodes

    def change_probs(self, node):
        node.change_probs()

    def __str__(self):
        return str([str(node) for node in self._nodes])


if __name__ == "__main__":
    g = Graph("simple_webservice")
    print(str(g))
