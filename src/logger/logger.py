# This source code is from the PyTorch Template Project (w/ slight adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/logger/logger.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
import json
import logging
import logging.config

from pathlib import Path
from collections import OrderedDict


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """Set up logging configuration."""
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config), file=sys.stderr)
        logging.basicConfig(level=default_level)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
