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


def get_null_graph():
    # type: () -> Graph
    return Graph("null", "null")


class Graph:
    def __init__(self, graph_name, attack_name):
        # type: (str, str) -> None
        """
        initializes an attack graph
        :param graph_name: name of the graph (in attack_graphs.json)
        :param attack_name: name of the attack (in attack_graphs.json)
        """
        self._graph_name = graph_name
        self._attack_name = attack_name

        graph_data = load_graph_data(graph_name)
        nodes_data = graph_data["nodes"]
        nodes_attack_data = graph_data["attacks"][attack_name]["nodes"]
        detection_systems_attack_data = graph_data["attacks"][attack_name]["detection_systems"]
        detection_system_data = graph_data["detection_systems"]

        self._nodes: List["n.Node"] = []
        self._detection_systems: List["d.DetectionSystem"] = []

        self._init_nodes(nodes_data)
        self._init_node_probs(nodes_attack_data)
        self._init_detection_systems(detection_system_data, detection_systems_attack_data)

    def _init_nodes(self, nodes_data):
        # type: (dict) -> None
        """
        parses the data from nodes_data and creates referenced nodes
        :param nodes_data: all data regarding nodes
        """

        # init nodes, init prev and next links as stings
        index = 0
        for node in nodes_data:
            self._nodes.append(n.Node(
                node,
                index,
                nodes_data[node]["previous"],
                nodes_data[node]["next"],
                nodes_data[node]["progress"],
                True if node.__contains__("honeypot") else False
            ))
            index += 1

        # set reference to previous node
        for node in self._nodes:
            prev_name = node.get_prev()
            for pot_node in self._nodes:
                if pot_node.get_name() == prev_name:
                    node.set_prev(pot_node)

        # set reference to next nodes
        for node in self._nodes:
            next_nodes = {}

            next_names = node.get_next()
            if next_names:

                for next_name in next_names:
                    for pot_node in self._nodes:
                        if pot_node.get_name() == next_name:
                            next_nodes[pot_node] = next_names[next_name]
                            break

            node.set_next(next_nodes)

    def _init_node_probs(self, attack_data):
        # type: (Dict[str, float]) -> None
        """
        looks up the probs in the nodes and sets numerical values according to attack data
        :param attack_data: name and value of all probs
        """
        used_keys = []

        # write parameters from json to nodes
        for node in self._nodes:

            next_nodes = node.get_next()
            for next_node in next_nodes:
                for prob in next_nodes[next_node]:

                    found_prob = False
                    for key in attack_data:
                        if next_nodes[next_node][prob] == key:
                            next_nodes[next_node][prob] = attack_data[key]
                            if not used_keys.__contains__(key):
                                used_keys.append(key)
                            found_prob = True
                            break

                    assert found_prob, f"\nNo key found with name: {next_nodes[next_node][prob]} in " \
                                       f"['{self._graph_name}']['{self._attack_name}'][nodes]." \
                                       f"\nThis key is needed for the transition: '{node.get_name()}' -> " \
                                       f"'{next_node.get_name()}'" \
                                       f"\nAdd for example '{next_nodes[next_node][prob]}': 0.1 to " \
                                       f"['{self._graph_name}']['{self._attack_name}'][nodes]."

            node.set_next(next_nodes)

        if len(attack_data) != len(used_keys):
            for key in used_keys:
                attack_data.pop(key)
            assert False, f"\nUnused keys from attack data: {str(attack_data)}." \
                          f"\nRemove those keys from ['{self._graph_name}'][attacks]['{self._attack_name}']"

        # add the current prob (equal to init)
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
        used_keys = []

        for detection_system in detection_system_data:

            # get probs
            probs = {}
            all_probs = detection_system_data[detection_system]["probs"]
            for prob in all_probs:

                found_prob = False
                for key in attack_data:
                    if all_probs[prob] == key:
                        probs[prob] = attack_data[key]
                        if not used_keys.__contains__(key):
                            used_keys.append(key)
                        found_prob = True
                        break

                location = f"['{self._graph_name}']['attacks']['{self._attack_name}']['detection_systems']."
                assert found_prob, f"\nNo key found with name: {all_probs[prob]} in {location} "\
                                   f"\nAdd for example '{all_probs[prob]}': 0.1 to {location}"

            # get reset node
            reset_node_name = detection_system_data[detection_system]["reset_node"]
            found_reset_node = False
            for node in self._nodes:
                if node.get_name() == reset_node_name:
                    reset_node_name = node
                    found_reset_node = True
                    break

            assert found_reset_node, f"\nNo node with name '{reset_node_name}' in nodes." \
                                     f"\nWrite a valid reset node to " \
                                     f"['{self._graph_name}']['detection_systems']['{detection_system}']['reset_node']."

            self._detection_systems.append(d.DetectionSystem(detection_system, probs, reset_node_name,
                                                             detection_system_data[detection_system]["after_nodes"]))

            # assign detection system to nodes
            nodes_names = detection_system_data[detection_system]["after_nodes"]
            for name in nodes_names:

                found_node = False
                for node in self._nodes:
                    if node.get_name() == name:
                        node.set_detection_system(self._detection_systems[-1])
                        found_node = True
                        break

                assert found_node, f"\nNo node with name '{name}' in nodes." \
                                   f"\nRemove or correct value '{name}' in " \
                                   f"['{self._graph_name}']['detection_systems']['{detection_system}']['after_nodes']."

        if len(attack_data) != len(used_keys):
            for key in used_keys:
                attack_data.pop(key)
            assert False, f"\nUnused keys from attack data: {str(attack_data)}." \
                          f"\nRemove those keys from ['{self._graph_name}'][attacks]['{self._attack_name}']"

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

    def get_obs_range(self):
        # type: () -> int
        return len(self._nodes) + 1

    def reset(self):
        # type: () -> None
        for node in self._nodes:
            node.reset()
        for detection_system in self._detection_systems:
            detection_system.reset()

    def __str__(self):
        return "Nodes: \n" + "\n".join([str(node) for node in self._nodes]) + \
               "\nDetection: \n" + "\n".join([str(detection) for detection in self._detection_systems])


if __name__ == "__main__":
    g = Graph("simple_webservice", "professional")
    print(g)
