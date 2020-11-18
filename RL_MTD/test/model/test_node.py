import unittest
import src.model.node as n


class NodeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = n.Node("tests", 0, None, {}, 0, False)
        self.next_node = n.Node("next", 1, self.node, {}, 1, False)
        self.another_next_node = n.Node("another_next", 2, self.node, {}, 1, False)

        self.node.set_next({self.next_node: {"init": 0.8, "dt": 0.01, "current": 0.8},
                            self.another_next_node: {"init": 0.7, "dt": 0.01, "current": 0.7}})

    def test_get_probs(self):
        new_probs = self.node.get_probs()
        self.assertEqual(list(new_probs.keys()), [self.next_node, self.another_next_node])
        self.assertEqual(list(new_probs.values()), [0.8, 0.7])

    def test_update_probs(self):
        self.node.update_probs()
        new_probs = self.node.get_probs()
        self.assertEqual(new_probs[self.next_node], 0.81)
        self.assertEqual(new_probs[self.another_next_node], 0.71)

    def test_reset_probs(self):
        # update
        self.node.update_probs()
        new_probs = self.node.get_probs()
        self.assertEqual(new_probs[self.next_node], 0.81)
        self.assertEqual(new_probs[self.another_next_node], 0.71)

        # reset
        self.node.reset_probs(self.next_node)
        self.node.reset_probs(self.another_next_node)
        new_probs = self.node.get_probs()
        self.assertEqual(new_probs[self.next_node], 0.80)
        self.assertEqual(new_probs[self.another_next_node], 0.70)

    def test_null_node(self):
        node = n.get_null_node()
        self.assertEqual(node.get_name(), "null")
        self.assertEqual(node.get_index(), 0)
        self.assertEqual(node.get_prev(), None)
        self.assertEqual(len(node.get_probs()), 0)
        self.assertEqual(node.get_detection_system(), None)
        self.assertEqual(node.get_progress_level(), 0)
        node.reset()


if __name__ == '__main__':
    unittest.main()
