import json

import src.model.node as n
import src.model.detection_system as d

from typing import List
from typing import Dict

from pathlib import Path
path = Path(__file__).parent / "../../config/attack_graphs.json"


def load_graph_data(graph_name):
    # type: (str) -> dict
    f = open(path, "r")
    all_data = json.load(f)
    f.close()
    return all_data[graph_name]


class Graph:
    def __init__(self, graph_name, attack_name):
        # type: (str, str) -> None
        """
        initializes an attack graph
        :param graph_name: name of the graph (in attack_graphs.json)
        :param attack_name: name of the attack (in attack_graphs.json)
        """
        graph_data = load_graph_data(graph_name)
        nodes_data = graph_data["nodes"]
        attack_data = graph_data["attacks"][attack_name]
        detection_system_data = graph_data["detection_systems"]

        self._nodes = []
        self._detection_systems = []
        self._min_progress_level = 1000
        self._max_progress_level = -1000

        self._init_nodes(nodes_data)
        self._init_node_probs(attack_data)
        self._init_detection_systems(detection_system_data, attack_data)

    def _init_nodes(self, nodes_data):
        # type: (dict) -> None
        """
        parses the data from nodes_data and creates referenced nodes
        :param nodes_data: all data regarding nodes
        """

        # init nodes, init prev and next links as stings
        for node in nodes_data:
            self._nodes.append(n.Node(
                node,
                nodes_data[node]["previous"],
                nodes_data[node]["next"],
                nodes_data[node]["progress"],
                True if node.__contains__("honeypot") else False
            ))

        # get min / max progression
        for node in self._nodes:
            progress_level = node.get_progress_level()

            if progress_level < self._min_progress_level:
                self._min_progress_level = progress_level

            if progress_level > self._max_progress_level:
                self._max_progress_level = progress_level

        assert self._min_progress_level != 1000, "Error, min progress not found!"
        assert self._max_progress_level != -1000, "Error, max progress not found!"

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

    def _init_node_probs(self, attack_data):
        # type: (Dict[str, float]) -> None
        """
        looks up the probs in the nodes and sets numerical values according to attack data
        :param attack_data: name and value of all probs
        """

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

        # add a current prob (equal to init)
        for node in self._nodes:
            for next_node in node.get_next():
                node.reset_probs(next_node)

    def _init_detection_systems(self, detection_system_data, attack_data):
        # type: (dict, dict) -> None
        """
        parses the nodes and probs from detection_system_data and attack_data, initializes detection systems and assigns
        them to the corresponding nodes
        :param detection_system_data: all data regarding a detection system
        :param attack_data: the probability of catching an attacker and change in probability after catching
        """

        for detection_system in detection_system_data:

            # get probs
            probs = {}
            for prob in detection_system_data[detection_system]["probs"]:

                found_prob = False
                for key in attack_data:
                    if detection_system_data[detection_system]["probs"][prob] == key:
                        probs[prob] = attack_data[key]
                        found_prob = True
                        break

                assert found_prob, f"Error, did not find prob {prob} for {detection_system}"

            reset_node_name = detection_system_data[detection_system]["reset_node"]
            found_reset_node = False
            for node in self._nodes:
                if node.get_name() == reset_node_name:
                    reset_node_name = node
                    found_reset_node = True
                    break

            assert found_reset_node, f"Error, did not find reset node {reset_node_name} for {detection_system}"

            self._detection_systems.append(d.DetectionSystem(detection_system, probs, reset_node_name))

            # assign detection system to nodes
            nodes_names = detection_system_data[detection_system]["after_nodes"]
            for name in nodes_names:

                found_node = False
                for node in self._nodes:
                    if node.get_name() == name:
                        node.set_detection_system(self._detection_systems[-1])
                        found_node = True
                        break

                assert found_node, f"Error, did not find node {name} for {detection_system}"

    def get_nodes(self):
        # type: () -> List[n.Node]
        """
        first node is always the start node, last is always the goal node
        :return: all nodes
        """
        return self._nodes

    def get_detection_systems(self):
        # type: () -> List["d.DetectionSystem"]
        return self._detection_systems

    def get_progress_levels_count(self):
        # type: () -> int
        return self._max_progress_level - self._min_progress_level + 1

    def __str__(self):
        return "Nodes: \n" + "\n".join([str(node) for node in self._nodes]) + \
               "\nDetection: \n" + "\n".join([str(detection) for detection in self._detection_systems])


if __name__ == "__main__":
    g = Graph("simple_webservice", "simple")
    print(g)
