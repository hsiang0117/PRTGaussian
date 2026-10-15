"""Focused state/atomic-write tests; no CUDA extensions are imported."""
import importlib
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runner = importlib.import_module('cloudgs_adapter.train_cloud')


class StateTests(unittest.TestCase):
    def test_rng_roundtrip(self):
        with patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), patch.object(torch.cuda, 'set_rng_state_all'):
            random.seed(91); np.random.seed(91); torch.manual_seed(91)
            state = runner.capture_rng()
            expected = (random.random(), np.random.rand(), torch.rand(3))
            runner.restore_rng(state)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_atomic_replacement_and_space_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'checkpoint_latest.pt'
            tensor = torch.ones(5)
            with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)):
                runner.atomic_checkpoint({'step': 1, 'value': tensor}, path)
                runner.atomic_checkpoint({'step': 2, 'value': tensor+1}, path)
            self.assertEqual(torch.load(path, weights_only=True)['step'], 2)
            self.assertFalse(path.with_name(path.name+'.partial').exists())
            with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaises(OSError):
                    runner.atomic_checkpoint({'step':3, 'value':tensor}, path)
            self.assertEqual(torch.load(path, weights_only=True)['step'], 2)


if __name__ == '__main__':
    unittest.main()
