import unittest
import src.model.node as n
import src.model.prevention_system as p


class PreventionSystemTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_node = n.Node("start", 0, None, {}, 1, False)
        self.node = n.Node("after", 1, self.start_node, {}, 1, False)
        self.honeypot = n.Node("honeypot", 2, self.node, {self.node: {"init": 0.4, "dt": 0.1}}, -1, True)

        self.prevention_system = p.PreventionSystem("IPS",
                                                    {"init": 0.8, "de": 0.5},
                                                    self.start_node,
                                                    ["start", "after"])

        self.node.set_prevention_system(self.prevention_system)
        self.honeypot.set_prevention_system(self.prevention_system)

    def test_get_prevention_system(self):
        self.assertEqual(self.node.get_prevention_system(), self.prevention_system)

    def test_get_probs(self):
        self.assertEqual(self.prevention_system.get_prob(), 0.8)

    def test_caught_attacker(self):
        reset_node = self.prevention_system.caught_attacker()
        self.assertEqual(self.start_node, reset_node)

    def test_reset(self):
        self.prevention_system.caught_attacker()
        self.prevention_system.reset_prob()
        self.assertEqual(self.prevention_system.get_prob(), 0.8)

    def test_update_probs(self):
        self.prevention_system.caught_attacker()
        self.assertEqual(self.prevention_system.get_prob(), 0.4)
        self.prevention_system.caught_attacker()
        self.assertEqual(self.prevention_system.get_prob(), 0.2)
        self.prevention_system.caught_attacker()
        self.assertEqual(self.prevention_system.get_prob(), 0.1)

    def test_learn(self):
        self.honeypot.get_prevention_system().learn()
        self.assertEqual(self.prevention_system.get_prob(), 0.9)
        self.honeypot.get_prevention_system().learn()
        self.assertEqual(self.prevention_system.get_prob(), 0.95)
        self.honeypot.get_prevention_system().learn()
        self.assertEqual(self.prevention_system.get_prob(), 0.975)

    def test_null_prevention_system(self):
        ds = d.get_null_prevention_system()
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
