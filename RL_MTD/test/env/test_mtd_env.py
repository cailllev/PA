import unittest

from src.env.mtd_env import MTDEnv, graph
from stable_baselines3.common.env_checker import check_env


def in_init_state(g):
    # type: ("graph.Graph") -> bool
    for node in g.get_nodes():
        for prob in node.get_next().values():
            if prob["init"] != prob["current"]:
                return False

    for detection_system in g.get_detection_systems():
        if detection_system.get_prob() != detection_system.get_init_prob():
            return False

    return True


class DetectionSystemTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env = MTDEnv()

    def test_in_init_state_method(self):
        self.assertTrue(in_init_state(graph))

    def test_check_env(self):
        check_env(self.env)

    def test_check_env_reset(self):
        self.env.reset()
        for _ in range(3):
            self.env.step([0, 0])

        self.assertFalse(in_init_state(graph))
        self.env.reset()
        self.assertTrue(in_init_state(graph))

    def test_pause_nodes(self):
        env = MTDEnv(nodes_pause=3)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 1])

    def test_pause_detection_systems(self):
        env = MTDEnv(detection_systems_pause=3)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 0])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 0])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 0])

    def test_pause_both(self):
        env = MTDEnv(nodes_pause=3, detection_systems_pause=3)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 1])
        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 0])

    def test_only_nodes(self):
        env = MTDEnv(only_nodes=True)
        env.reset()

        self.assertRaises(Exception, env.step, [1, 1])
        self.assertRaises(Exception, env.step, [0, 1])
        env.step([1, 0])
        env.step([0, 0])
        self.assertTrue(True)

    def test_only_detection_systems(self):
        env = MTDEnv(only_detection_systems=True)
        env.reset()

        self.assertRaises(Exception, env.step, [1, 1])
        self.assertRaises(Exception, env.step, [1, 0])
        env.step([0, 1])
        env.step([0, 0])
        self.assertTrue(True)

    # disable both node restarts and detection system switches
    def test_static(self):
        env = MTDEnv(only_nodes=True, only_detection_systems=True)
        env.reset()

        self.assertRaises(Exception, env.step, [1, 1])
        self.assertRaises(Exception, env.step, [1, 0])
        self.assertRaises(Exception, env.step, [0, 1])
        env.step([0, 0])
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
