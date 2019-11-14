import unittest

import torch


class TestAddDict(unittest.TestCase):

    def test1(self):
        a = {"a": torch.tensor(0.), "b": torch.tensor(1.)}
        b = {"a": torch.tensor(5.), "b": torch.tensor(1.)}
        self.assertEqual(add_dict(a, b), b)


if __name__ == '__main__':
    unittest.main()
