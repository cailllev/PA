import json
from src.model.node import *


def load_json(graph_name):
    f = open("../../config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[graph_name]


def load_nodes(nodes_data):
    nodes = []
    for key in nodes_data:
        n = NodeData(key, nodes_data[key])
        nodes.append(n)
    return nodes


def load_attack(graph_data, attack_name):
    return graph_data["attacks"][attack_name]


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
        next_nodes = {}

        next_names = node.get_next()
        if next_names:

            for next_name in next_names:
                for pot_node in nodes:
                    if pot_node.get_name() == next_name:
                        next_nodes[pot_node] = next_names[next_name]
                        test = 1

        node.set_next(next_nodes)

    return nodes


class Graph:
    def __init__(self, graph_name, attack_name):
        graph_data = load_json(graph_name)

        nodes_data = load_nodes(graph_data["nodes"])
        self._nodes = init_graph(nodes_data)
        self.init_probs(graph_data, attack_name)

    def init_probs(self, graph_data, name_attack):
        attack_data = load_attack(graph_data, name_attack)

        # write parameters from json to nodes
        for node in self._nodes:
            next_nodes = node.get_next()
            for next_node in next_nodes:
                for param in next_nodes[next_node]:
                    for key in attack_data:
                        if next_nodes[next_node][param] == key:
                            next_nodes[next_node][param] = attack_data[key]
                            break
            node.set_next(next_nodes)

        # add a current value to each edge
        for node in self._nodes:
            next_nodes = node.get_next()
            for next_node in next_nodes:
                next_nodes[next_node]["current"] = next_nodes[next_node]["init"]
            node.set_next(next_nodes)

    def init_services(self):
        pass

    def get_nodes(self):
        """
        first node is always the start node, last is always the goal node
        :return: all nodes
        """
        return self._nodes

    def __str__(self):
        return str([str(node) for node in self._nodes])


if __name__ == "__main__":
    g = Graph("simple_webservice", "simple")
    print(g)
