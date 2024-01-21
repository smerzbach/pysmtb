import tempfile
from torch.utils.tensorboard import SummaryWriter
import unittest
from pysmtb.tb.tblogparser import TensorboardLogParser


class TestTBLogParser(unittest.TestCase):
    def setUp(self) -> None:
        # initialize by creating a dummy log file using SummaryWriter
        self.temp_dir = tempfile.mkdtemp()
        writer = SummaryWriter(log_dir=self.temp_dir)

        max_epochs = 10000
        for ep in range(max_epochs):
            writer.add_scalar('loss/train', 1.0 / (ep + 1), ep)
            if ep % 100 == 0:
                writer.add_scalar('loss/validation', 1.0 - 1.0 / (ep + 1), ep)
        writer.add_scalar('metrics/accuracy', 0.5, max_epochs)
        writer.close()

    def tearDown(self) -> None:
        # remove temp dir
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_tb_log_parser(self):
        # parse directly from TB events file and compare to expected results
        tb = TensorboardLogParser(log_dir=self.temp_dir, caching=True, verbose=False)
        self.assertEqual(tb.scalar_names, ['loss/train', 'loss/validation', 'metrics/accuracy'])

        # parse a second time from cache file
        tb2 = TensorboardLogParser(log_dir=self.temp_dir, caching=True, verbose=False)
        self.assertEqual(tb.scalar_names, tb2.scalar_names)

        # compare extracted dataframes (ignoring any entries that are NaN)
        self.assertTrue((tb.df.fillna(0) == tb2.df.fillna(0)).all().all())

        # test resetting cache file
        tb3 = TensorboardLogParser(log_dir=self.temp_dir, caching=True, verbose=False, reset_cache=True)
        self.assertTrue(tb3.acc is not None)
        self.assertTrue(tb3.cache_file.exists())


if __name__ == '__main__':
    unittest.main()
