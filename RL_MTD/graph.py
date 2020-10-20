import json
from node import *


def load_json(name_graph):
    f = open("config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[name_graph]


def load_nodes(nodes_data):
    nodes = []
    for key in nodes_data:
        n = NodeData(key, nodes_data[key])
        nodes.append(n)
    return nodes


class NodeData:
    def __init__(self, name, data):
        self.name = name
        self.prev = data["previous"]
        self.next = data["next"]


def init_graph(nodes_data):
    nodes = []

    # init nodes and previous nodes
    for node in nodes_data:
        n = Node(node.name, node.prev)
        nodes.append(n)

    # set reference to next nodes
    for node in nodes:
        name = node.get_name()
        # get the next nodes of current (via prev == name)
        for next_node in nodes:
            if next_node.get_prev() == name:
                node.set_next(next_node)

    # set delta t to each edge (changes over time)
    for node in nodes:
        for node_data in nodes_data:
            pass

    return nodes


class Graph:
    def __init__(self, name_graph):
        graph_data = load_json(name_graph)

        nodes_data = load_nodes(graph_data["nodes"])
        self._nodes = init_graph(nodes_data)

        self._start_node = self._nodes[0]
        self._end_node = self._nodes[-1]


if __name__ == "__main__":
    g = Graph("simple_webservice")
