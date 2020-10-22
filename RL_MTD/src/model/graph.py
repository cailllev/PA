import json
import src.model.node as n


def load_json(graph_name):
    f = open("../../config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[graph_name]


class Graph:
    def __init__(self, graph_name, attack_name):
        graph_data = load_json(graph_name)

        self._nodes = []
        self.init_nodes(graph_data)
        self.init_probs(graph_data, attack_name)
        self.init_services(graph_data)

    def init_nodes(self, graph_data):
        # init nodes, init prev and next links as stings
        nodes = graph_data["nodes"]
        for node in nodes:
            self._nodes.append(n.Node(
                node,
                nodes[node]["previous"],
                nodes[node]["next"]
            ))

        # set reference to previous node
        for node in self._nodes:
            prev_name = node.get_prev()
            for pot_node in self._nodes:
                if pot_node.get_name() == prev_name:
                    node.set_prev(pot_node)

        # set reference and probability to next nodes
        for node in self._nodes:
            next_nodes = {}

            next_names = node.get_next()
            if next_names:

                for next_name in next_names:
                    for pot_node in self._nodes:
                        if pot_node.get_name() == next_name:
                            next_nodes[pot_node] = next_names[next_name]

            node.set_next(next_nodes)

    def init_probs(self, graph_data, name_attack):
        attack_data = graph_data["attacks"][name_attack]

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

    def init_services(self, graph_data):
        # TODO
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
