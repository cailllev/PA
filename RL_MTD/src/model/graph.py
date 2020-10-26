import json
import src.model.node as n
import src.model.detection_system as d


def load_graph_data(graph_name):
    f = open("../../config/attack_graphs.json", "r")
    all_data = json.load(f)
    return all_data[graph_name]


class Graph:
    def __init__(self, graph_name, attack_name):
        graph_data = load_graph_data(graph_name)
        attack_data = graph_data["attacks"][attack_name]

        self._nodes = []
        self._restartable_nodes = []
        self._detection_systems = []

        self.init_nodes(graph_data)
        self.init_probs(attack_data)
        self.init_detection_systems(graph_data, attack_data)

    def init_nodes(self, graph_data):
        # init nodes, init prev and next links as stings
        nodes = graph_data["nodes"]
        for node in nodes:
            self._nodes.append(n.Node(
                node,
                nodes[node]["previous"],
                nodes[node]["next"]
            ))
            if nodes[node]["restartable"]:
                self._restartable_nodes.append(self._nodes[-1])

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

    def init_probs(self, attack_data):
        # write parameters from json to nodes
        for node in self._nodes:

            next_nodes = node.get_next()
            for next_node in next_nodes:
                for prob in next_nodes[next_node]:

                    found_prob = False
                    for key in attack_data:
                        if next_nodes[next_node][prob] == key:
                            next_nodes[next_node][prob] = attack_data[key]
                            found_prob = True
                            break

                    assert found_prob, f"Error, did not find prob {prob} for {next_node.get_name()}"

            node.set_next(next_nodes)

        # add a current value to each edge
        for node in self._nodes:
            next_nodes = node.get_next()
            for next_node in next_nodes:
                next_nodes[next_node]["current"] = next_nodes[next_node]["init"]
            node.set_next(next_nodes)

    def init_detection_systems(self, graph_data, attack_data):

        detection_systems_types = graph_data["detection_systems"]
        for detection_system in detection_systems_types:

            # get probs
            probs = {}
            for prob in detection_systems_types[detection_system]["probs"]:

                found_prob = False
                for key in attack_data:
                    if detection_systems_types[detection_system]["probs"][prob] == key:
                        probs[prob] = attack_data[key]
                        found_prob = True
                        break

                assert found_prob, f"Error, did not find prob {prob} for {detection_system}"

            reset_node_name = detection_systems_types[detection_system]["reset_node"]
            found_reset_node = False
            for node in self._nodes:
                if node.get_name() == reset_node_name:
                    reset_node_name = node
                    found_reset_node = True
                    break

            assert found_reset_node, f"Error, did not find reset node {reset_node_name} for {detection_system}"

            self._detection_systems.append(d.DetectionSystem(detection_system, probs, reset_node_name))

            # assign detection system to nodes
            nodes_names = detection_systems_types[detection_system]["after_nodes"]
            for name in nodes_names:

                found_node = False
                for node in self._nodes:
                    if node.get_name() == name:
                        node.set_detection_system(self._detection_systems[-1])
                        found_node = True
                        break

                assert found_node, f"Error, did not find node {name} for {detection_system}"

    def get_nodes(self):
        """
        first node is always the start node, last is always the goal node
        :return: all nodes
        """
        return self._nodes

    def get_detection_systems(self):
        return self._detection_systems

    def get_restartable_nodes_count(self):
        return len(self._restartable_nodes)

    def get_detection_systems_count(self):
        return len(self._detection_systems)

    def __str__(self):
        return "Nodes: " + str([str(node) for node in self._nodes]) + \
               "Detection: " + str([str(detection) for detection in self._detection_systems])


if __name__ == "__main__":
    g = Graph("simple_webservice", "simple")
    print(g)
