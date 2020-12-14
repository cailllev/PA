import unittest

import src.defender2000 as d
from src.env.mtd_env import set_config


class Defender2000Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_config("simple_webservice", "professional")
        self._defender2000 = d.Defender2000()

    def test_get_index(self):
        index = d.get_prevention_system_action_index_from_obs(1)
        self.assertEqual(index, 1)

    def test_get_predict_simple(self):
        self._defender2000.reset()

        incident_point = 1
        pred = self._defender2000.predict(incident_point)[0]
        index = d.get_prevention_system_action_index_from_obs(incident_point)
        self.assertEqual(pred, [incident_point, index])

    def test_get_predict_advanced(self):
        """
        obs: 3, 0 -> actions: [3, 1], [1, ?]
        """
        self._defender2000.reset()

        incident_point = 3  # 3=planner <- 1=authorizer <- 0=internet

        pred = self._defender2000.predict(incident_point)[0]
        index = d.get_prevention_system_action_index_from_obs(incident_point)
        self.assertEqual(pred, [incident_point, index])

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 1)

    def test_get_predict_advanced2(self):
        """
        obs: 5, 0, 0 -> actions: [5, 1], [3, ?], [1, ?]
        """
        self._defender2000.reset()

        incident_point = 5  # 5=db1 <- 3=planner <- 1=authorizer <- 0=internet

        pred = self._defender2000.predict(incident_point)[0]
        index = d.get_prevention_system_action_index_from_obs(incident_point)
        self.assertEqual(pred, [incident_point, index])

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 3)

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 1)

    def test_predict_with_pause(self):
        """
        pause: 3,2
        obs: 1, 1, 1, 1, 1 -> actions: [1, 1], [!1, !1], [!1, 1], [1, !1], [!1, 1]
        """
        self._defender2000 = d.Defender2000(False, False, 3, 2)
        self._defender2000.reset()

        pred = self._defender2000.predict(1)[0]
        self.assertEqual(pred, [1, 1])

        pred = self._defender2000.predict(1)[0]
        self.assertNotEqual(pred[0], 1)
        self.assertNotEqual(pred[1], 1)

        pred = self._defender2000.predict(1)[0]
        self.assertNotEqual(pred[0], 1)
        self.assertEqual(pred[1], 1)

        pred = self._defender2000.predict(1)[0]
        self.assertEqual(pred[0], 1)
        self.assertNotEqual(pred[1], 1)

        pred = self._defender2000.predict(1)[0]
        self.assertNotEqual(pred[0], 1)
        self.assertEqual(pred[1], 1)

    def test_get_predict_deeper_in(self):
        """
        obs: 3, 5 -> actions: [3, 1], [5, 2], [3, ?], [1, ?]
        """
        self._defender2000.reset()

        pred = self._defender2000.predict(3)[0]
        self.assertEqual(pred, [3, 1])

        pred = self._defender2000.predict(5)[0]
        self.assertEqual(pred, [5, 2])

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 3)

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 1)

    def test_get_predict_deeper_in2(self):
        """
        obs: 3, 5 -> actions: [5, 2], [3, ?], [5, 2], [3, ?], [1, ?]
        """
        self._defender2000.reset()

        pred = self._defender2000.predict(5)[0]
        self.assertEqual(pred, [5, 2])

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 3)

        pred = self._defender2000.predict(5)[0]
        self.assertEqual(pred, [5, 2])

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 3)

        pred = self._defender2000.predict(0)[0]
        self.assertEqual(pred[0], 1)


if __name__ == '__main__':
    unittest.main()
