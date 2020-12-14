import unittest

import src.env.mtd_env as m
from stable_baselines3.common.env_checker import check_env


def in_init_state(g):
    # type: ("graph.Graph") -> bool
    for node in g.get_nodes():
        for prob in node.get_next().values():
            if prob["init"] != prob["current"]:
                return False

    for prevention_system in g.get_prevention_systems():
        if prevention_system.get_prob() != prevention_system.get_init_prob():
            return False

    return True


class MTDEnvTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # maybe not best idea to check this in __init__
        # -> works outside __init__ -> but then creating self.env also outside __init__ -> then self.env is None outside
        # __init__ -> Errors
        # self.assertRaises(Exception, m.MTDEnv)

        m.set_config("simple_webservice", "professional")
        self.env = m.MTDEnv()

    def test_in_init_state_method(self):
        self.assertTrue(in_init_state(m.graph))

    def test_check_env(self):
        check_env(self.env)

    def test_check_env_reset(self):
        self.env.reset()
        for _ in range(3):
            self.env.step([0, 0])

        self.assertFalse(in_init_state(m.graph))
        self.env.reset()
        self.assertTrue(in_init_state(m.graph))

    def test_pause_nodes(self):
        env = m.MTDEnv(nodes_pause=3)
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

    def test_pause_prevention_systems(self):
        env = m.MTDEnv(prevention_systems_pause=3)
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
        env = m.MTDEnv(nodes_pause=3, prevention_systems_pause=3)
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
        env = m.MTDEnv(only_nodes=True)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [1, 0])
        env.reset()
        env.step([1, 0])
        self.assertEqual(env.get_last_action(), [1, 0])
        env.reset()
        env.step([0, 1])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()
        env.step([0, 0])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()

    def test_only_prevention_systems(self):
        env = m.MTDEnv(only_prevention_systems=True)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 1])
        env.reset()
        env.step([1, 0])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()
        env.step([0, 1])
        self.assertEqual(env.get_last_action(), [0, 1])
        env.reset()
        env.step([0, 0])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()

    # disable both node restarts and prevention system switches
    def test_static(self):
        env = m.MTDEnv(only_nodes=True, only_prevention_systems=True)
        env.reset()

        env.step([1, 1])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()
        env.step([1, 0])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()
        env.step([0, 1])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()
        env.step([0, 0])
        self.assertEqual(env.get_last_action(), [0, 0])
        env.reset()


if __name__ == '__main__':
    unittest.main()
