import unittest

from src.env.mtd_env import MTDEnv
from stable_baselines3.common.env_checker import check_env


class DetectionSystemTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env = MTDEnv()

    def test_check_env(self):
        check_env(self.env)


if __name__ == '__main__':
    unittest.main()
