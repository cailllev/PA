import json

import src.model.node as n
import src.model.prevention_system as p

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
        prevention_systems_attack_data = graph_data["attacks"][attack_name]["prevention_systems"]
        prevention_system_data = graph_data["prevention_systems"]

        self._nodes: List["n.Node"] = []
        self._prevention_systems: List["p.PreventionSystem"] = []

        self._init_nodes(nodes_data)
        self._init_node_probs(nodes_attack_data)
        self._init_prevention_systems(prevention_system_data, prevention_systems_attack_data)

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

    def _init_prevention_systems(self, prevention_system_data, attack_data):
        # type: (dict, dict) -> None
        """
        parses the nodes and probs from prevention_system_data and attack_data, initializes prevention systems and
        assigns them to the corresponding nodes
        :param prevention_system_data: all data regarding a prevention system
        :param attack_data: the probability of catching an attacker and change in probability after catching
        """
        used_keys = []

        for prevention_system in prevention_system_data:

            # get probs
            probs = {}
            all_probs = prevention_system_data[prevention_system]["probs"]
            for prob in all_probs:

                found_prob = False
                for key in attack_data:
                    if all_probs[prob] == key:
                        probs[prob] = attack_data[key]
                        if not used_keys.__contains__(key):
                            used_keys.append(key)
                        found_prob = True
                        break

                location = f"['{self._graph_name}']['attacks']['{self._attack_name}']['prevention_systems']."
                assert found_prob, f"\nNo key found with name: {all_probs[prob]} in {location} "\
                                   f"\nAdd for example '{all_probs[prob]}': 0.1 to {location}"

            # get reset node
            reset_node_name = prevention_system_data[prevention_system]["reset_node"]
            found_reset_node = False
            for node in self._nodes:
                if node.get_name() == reset_node_name:
                    reset_node_name = node
                    found_reset_node = True
                    break

            assert found_reset_node, f"\nNo node with name '{reset_node_name}' in nodes." \
                                     f"\nWrite a valid reset node to " \
                                     f"['{self._graph_name}']['prevention_systems']" \
                                     f"['{prevention_system}']['reset_node']."

            self._prevention_systems.append(p.PreventionSystem(prevention_system, probs, reset_node_name,
                                                               prevention_system_data[prevention_system]
                                                               ["after_nodes"]))

            # assign prevention system to nodes
            nodes_names = prevention_system_data[prevention_system]["after_nodes"]
            for name in nodes_names:

                found_node = False
                for node in self._nodes:
                    if node.get_name() == name:
                        node.set_prevention_system(self._prevention_systems[-1])
                        found_node = True
                        break

                assert found_node, f"\nNo node with name '{name}' in nodes." \
                                   f"\nRemove or correct value '{name}' in " \
                                   f"['{self._graph_name}']" \
                                   f"['prevention_systems']['{prevention_system}']['after_nodes']."

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

    def get_prevention_systems(self):
        # type: () -> List["p.PreventionSystem"]
        return self._prevention_systems

    def get_obs_range(self):
        # type: () -> int
        return len(self._nodes) + 1

    def reset(self):
        # type: () -> None
        for node in self._nodes:
            node.reset()
        for prevention_system in self._prevention_systems:
            prevention_system.reset()

    def __str__(self):
        return "Nodes: \n" + "\n".join([str(node) for node in self._nodes]) + \
               "\nDetection: \n" + "\n".join([str(detection) for detection in self._prevention_systems])


if __name__ == "__main__":
    g = Graph("simple_webservice", "professional")
    print(g)
