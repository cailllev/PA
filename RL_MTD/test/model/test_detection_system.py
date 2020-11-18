import unittest
import src.model.node as n
import src.model.detection_system as d


class DetectionSystemTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_node = n.Node("start", 0, None, {}, 1, False)
        self.node = n.Node("after", 1, self.start_node, {}, 1, False)
        self.honeypot = n.Node("honeypot", 2, self.node, {self.node: {"init": 0.4, "dt": 0.1}}, -1, True)

        self.detection_system = d.DetectionSystem("IDS", {"init": 0.8, "de": 0.5}, self.start_node, ["start", "after"])

        self.node.set_detection_system(self.detection_system)
        self.honeypot.set_detection_system(self.detection_system)

    def test_get_detection_system(self):
        self.assertEqual(self.node.get_detection_system(), self.detection_system)

    def test_get_probs(self):
        self.assertEqual(self.detection_system.get_prob(), 0.8)

    def test_caught_attacker(self):
        reset_node = self.detection_system.caught_attacker()
        self.assertEqual(self.start_node, reset_node)

    def test_reset(self):
        self.detection_system.caught_attacker()
        self.detection_system.reset_prob()
        self.assertEqual(self.detection_system.get_prob(), 0.8)

    def test_update_probs(self):
        self.detection_system.caught_attacker()
        self.assertEqual(self.detection_system.get_prob(), 0.4)
        self.detection_system.caught_attacker()
        self.assertEqual(self.detection_system.get_prob(), 0.2)
        self.detection_system.caught_attacker()
        self.assertEqual(self.detection_system.get_prob(), 0.1)

    def test_learn(self):
        self.honeypot.get_detection_system().learn()
        self.assertEqual(self.detection_system.get_prob(), 0.9)
        self.honeypot.get_detection_system().learn()
        self.assertEqual(self.detection_system.get_prob(), 0.95)
        self.honeypot.get_detection_system().learn()
        self.assertEqual(self.detection_system.get_prob(), 0.975)

    def test_null_detection_system(self):
        ds = d.get_null_detection_system()
        self.assertEqual(ds.get_name(), "null")
        self.assertEqual(ds.caught_attacker().get_name(), n.get_null_node().get_name())

        # no learning with null
        self.assertEqual(ds.get_prob(), 0)
        ds.reset()
        self.assertEqual(ds.get_prob(), 0)
        ds.learn()
        self.assertEqual(ds.get_prob(), 0)


if __name__ == '__main__':
    unittest.main()
