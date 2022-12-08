import itertools
import unittest
from portiloop.src.demo.offline import run_offline
from pathlib import Path


class TestOffline(unittest.TestCase):

    def setUp(self):
        combinatorial_config = {
            'offline_filtering': [True, False],
            'online_filtering': [True, False],
            'online_detection': [True, False],
            'wamsley': [True, False],
            'lacourse': [True, False],
        }

        self.exclusives = [("duplicate_as_window", "use_cnn_encoder")]

        keys = list(combinatorial_config)
        all_options_iterator = itertools.product(*map(combinatorial_config.get, keys))
        all_options_dicts = [dict(zip(keys, values)) for values in all_options_iterator]
        self.filtered_options = [value for value in all_options_dicts if (value['online_detection'] and value['online_filtering']) or not value['online_detection']]
        self.xdf_file = Path(__file__).parents[3] / "test_xdf" / "test_file.xdf"


    def test_all_options(self):
        for config in self.filtered_options:
            if config['online_detection']:
                self.assertTrue(config['online_filtering'])

    def test_single_option(self):
        res = list(run_offline(
                self.xdf_file,
                offline_filtering=True,
                online_filtering=True,
                online_detection=True,
                wamsley=True,
                lacourse=True,
                threshold=0.5,
                channel_num=2,
                freq=250))
        print(res)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()