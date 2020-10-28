import unittest
import src.model.node as n
import src.model.detection_system as d


class DetectionSystemTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_node = n.Node("null", None, {}, 1, False)
        self.node = n.Node("start", self.start_node, {}, 1, False)
        self.honeypot = n.Node("honeypot", self.node, {self.node: {"init": 0.4, "dt": 0.1}}, -1, True)

        self.detection_system = d.DetectionSystem("IDS", {"init": 0.8, "de": 0.5}, self.start_node)

        self.node.set_detection_system(self.detection_system)
        self.honeypot.set_detection_system(self.detection_system)

    def test_get_detection_system(self):
        self.assertEqual(self.node.get_detection_system(), self.detection_system)

    def test_get_probs(self):
        self.assertEqual(self.detection_system.get_prob(), 0.8)

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


if __name__ == '__main__':
    unittest.main()
