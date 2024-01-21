import numpy as np
import pandas as pd
from pathlib import Path
import re
from tensorboard.backend.event_processing import event_accumulator as ea
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tqdm import tqdm
from typing import *


class TensorboardLogParser:
    _peek_max_start_entries = 2000
    _peek_max_end_entries = 1000

    # currently only events.* files are loaded which end in .0
    _valid_log_suffix = '.0'

    def __init__(
            self,
            log_dir: Union[str, Path],
            pattern: Optional[str] = None,
            exclude_pattern: Optional[str] = None,
            caching: bool = True,
            reset_cache: bool = False,
            verbose: bool = False,
    ):
        """
        Extract all scalar values from a tensorboard log file and store them in a pandas dataframe.
        With the optional regular expressions pattern and exclude_pattern, only a subset of the scalar values can be
        selected for extraction. The dataframe has the scalar names as columns and the step values as index. The
        dataframe is stored in the attribute df.

        Parsing big log files can take a long time. If caching == True, the dataframe is cached as a pickle file in the
        same directory as the log file. This allows much faster loading of the dataframes in subsequent calls. The cache
        file is only updated if the log file has been modified since the last call. With reset_cache == True, the cache
        file deletion is enforced and the log file is parsed again.

        :param log_dir: path to directory containing tensorboard log files
        :param pattern: regex pattern to match scalar names
        :param exclude_pattern: regex pattern to exclude scalar names
        :param caching: whether to cache parsed dataframes
        :param reset_cache: whether to delete cache file and force parsing of log file
        :param verbose: whether to print progress
        """
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise ValueError('log_dir %s does not exist' % self.log_dir)
        if not self.log_dir.is_dir():
            raise ValueError('log_dir %s must be a directory' % self.log_dir)
        self.log_file = None
        self.acc = None
        self.df = None

        self.pattern = pattern
        self.exclude_pattern = exclude_pattern

        self.caching = caching
        self.verbose = verbose

        self._scalar_names = None

        if reset_cache:
            self._get_log_file()
            self.cache_file.unlink(missing_ok=True)

        self._parse_log_dir()

    @property
    def cache_file(self) -> Path:
        return Path(str(self.log_file) + '.pkl')

    @property
    def scalar_names(self) -> List[str]:
        if self.acc:
            return self.acc.Tags()['scalars']
        else:
            return self._peek_log_file()

    def _get_log_file(self):
        log_files = [f for f in self.log_dir.iterdir() if
                     f.is_file() and f.name.startswith('events.') and f.name.endswith(self._valid_log_suffix)]
        if len(log_files) == 0:
            raise ValueError('log_dir %s does not contain any log files' % self.log_dir)
        if len(log_files) > 1:
            raise ValueError('log_dir %s contains multiple log files' % self.log_dir)
        self.log_file = log_files[0]

    def _parse_log_dir(self):
        self._get_log_file()
        # check if pkl cache file exists with a timestamp newer than the log file
        if not self._validate_cache():
            self._load_from_log()

    def _peek_log_file(self) -> List[str]:
        if self._scalar_names is None:
            event_gen = RawEventFileLoader(str(self.log_file)).Load()
            keys = set()
            counter = 0
            for event in tqdm(event_gen, desc='peeking log file keys', disable=not self.verbose):
                counter += 1
                if counter > self._peek_max_start_entries:
                    continue
                event = event_pb2.Event.FromString(event)
                if event.HasField('summary'):
                    for value in event.summary.value:
                        if value.tag != 'custom_scalars__config__':
                            keys.add(value.tag)
            # second run to get 1000 last entries (metrics are usually only dumped after training at the end)
            event_gen = RawEventFileLoader(str(self.log_file)).Load()
            length = counter
            counter = 0
            for event in tqdm(event_gen, desc='peeking log file keys', disable=not self.verbose):
                counter += 1
                if counter >= length - self._peek_max_end_entries:
                    event = event_pb2.Event.FromString(event)
                    if event.HasField('summary'):
                        for value in event.summary.value:
                            if value.tag != 'custom_scalars__config__':
                                keys.add(value.tag)
            self._scalar_names = sorted(list(keys))
        return self._scalar_names

    def _validate_cache(self) -> bool:
        valid = self.cache_file.exists() and self.cache_file.stat().st_mtime > self.log_file.stat().st_mtime

        # check if all scalar names matching pattern in the log file are also present in the cache
        if valid:
            self._load_from_cache()
            for tag in self.scalar_names:
                if self.pattern is not None and re.match(self.pattern, tag) is None:
                    continue
                if tag not in self.df.columns:
                    valid = False
                    break
        return valid

    def _load_from_cache(self):
        if self.verbose:
            print('loading cache from %s' % self.cache_file)
        self.df = pd.read_pickle(self.cache_file)
    
    def _load_from_log(self):
        if self.verbose:
            print('parsing log file %s' % self.log_file)

        self.df = None
        self.acc = ea.EventAccumulator(str(self.log_file))
        self.acc.Reload()
        tags_with_single_entry = []
        for tag in tqdm(self.scalar_names, desc='parsing log file', disable=not self.verbose):
            if self.pattern is not None and re.match(self.pattern, tag) is None:
                continue
            if self.exclude_pattern is not None and re.match(self.exclude_pattern, tag) is not None:
                continue
            steps = []
            values = []
            for s in self.acc.Scalars(tag):
                steps.append(int(s.step))
                values.append(s.value)

            if len(steps) == 1:
                tags_with_single_entry.append(tag)

            df = pd.DataFrame(data=np.array(values, dtype=np.float32), index=np.array(steps, dtype=np.int32),
                              columns=[tag])
            if self.df is None:
                self.df = df
            else:
                self.df = self.df.join(df, how='outer')

        # replicate single valid entry to all NaNs
        for tag in tags_with_single_entry:
            vals = self.df[tag].values
            valid_value = vals[~np.isnan(vals)][0]
            vals[np.isnan(vals)] = valid_value

        # tag with log path
        self.df.attrs['log_path'] = self.acc.path
        
        if self.caching:
            self.df.to_pickle(self.cache_file)
