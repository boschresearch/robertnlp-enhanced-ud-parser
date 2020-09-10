# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import argparse

from init_config import ConfigParser


def main(config):
    """Initialize model, data loaders and trainer based on config file and run training."""
    model = config.init_model()

    data_loaders = config.init_data_loaders(model)

    trainer = config.init_trainer(model, data_loaders["train"], data_loaders["dev"])

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Graph-based enhanced UD parser (training mode)')
    args.add_argument('config', type=str, help='config file path (required)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
