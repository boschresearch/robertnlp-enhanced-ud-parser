# This source code is from the PyTorch Template Project (w/ heavy adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/parse_config.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import logging
import json

import data_handling.vocab as vocab_module
import data_handling.data_loaders as data_loaders_module
import torch.optim as optimizers_module
import torch.nn.modules.loss as loss_module
import models.label_scorers as models_module
import models.embeddings.wrappers as embeddings_module
import models.biaffine as classifiers_module

from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from collections import OrderedDict

from logger.logger import setup_logging
from trainer.trainer import Trainer
from models.embeddings.embeddings_processor import EmbeddingsProcessor


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """This class parses the configuration json file and handles hyperparameters for training, checkpoint saving
        and logging module.

        Most importantly, however, it initializes the actual model itself, including all of its components (e.g. the
        embedding layer or the biaffine classifier). This means that this class actually does a lot of the "heavy
        lifting" that you might expect to find in the constructor of the model itself. However, I've decided to put it
        here in order to keep that class readable at a glance.

        Parameters:
        config: Dict containing configurations, hyperparameters for training (contents of 'config.json' file).
        resume: Path to the checkpoint being loaded.
        modification: Dict keychain:value, specifying position values to be replaced from config dict.
        run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
                Timestamp is being used as default.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        experiment_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / experiment_name / run_id
        self._log_dir = save_dir / 'log' / experiment_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def init_model(self):
        """Initialize the model as specified in the configuration."""
        params = self["model"]
        model_type = params["type"]
        model_args = params["args"]

        # Initialise all the components of the model:
        # Embeddings -> -> FC layer for heads/dependents -> Classifier
        model_args["embeddings_processor"] = self._init_embeddings(model_args["embeddings_processor"])
        embeddings_dim = model_args["embeddings_processor"].output_dim

        model_args["output_vocab"] = self._init_vocab(model_args["output_vocab"])
        output_dim = len(model_args["output_vocab"])

        model_args["classifier"] = self._init_classifier(model_args["classifier"], embeddings_dim, output_dim)

        # Build and return the actual model
        return getattr(models_module, model_type)(**model_args)

    def _init_embeddings(self, params):
        input_embeddings = list()
        for input_embedding in params["inputs"]:
            input_embedding = getattr(embeddings_module, input_embedding["type"])(**input_embedding["args"])
            input_embeddings.append(input_embedding)

        embeddings_args = params["args"]
        embeddings_args["input_embeddings"] = input_embeddings

        return EmbeddingsProcessor(**embeddings_args)

    def _init_classifier(self, params, input_dim, output_dim):
        classifier_type = params["type"]
        classifier_args = params["args"]

        classifier_args["input1_size"] = input_dim
        classifier_args["input2_size"] = input_dim
        classifier_args["output_size"] = output_dim

        return getattr(classifiers_module, classifier_type)(**classifier_args)

    def _init_vocab(self, params):
        vocab_type = params["type"]
        vocab_args = params["args"]

        return getattr(vocab_module, vocab_type)(**vocab_args)

    def init_data_loaders(self, model):
        """Initialize the data loaders as specified in the configuration file, and in such a way that they provide
        valid input for the given model"""
        params = self["data_loaders"]
        data_loader_type = params["type"]
        data_loader_args = params["args"]

        data_loader_args["label_vocab"] = model.output_vocab

        data_loaders = dict()
        for p in params["paths"]:
            data_loader_args["corpus_path"] = params["paths"][p]
            data_loaders[p] = getattr(data_loaders_module, data_loader_type)(**data_loader_args)

        return data_loaders

    def init_trainer(self, model, train_data_loader, dev_data_loader):
        """Initialize the trainer for the given model. The model is trained on the specified train_data_loader and
        validated on the specified dev_data_loader."""
        params = self["trainer"]

        optimizer = self._init_optimizer(model, params["optimizer"])

        criterion = self._init_criterion(params["loss"])

        trainer = Trainer(model, self, optimizer, criterion, train_data_loader, dev_data_loader)

        return trainer

    def _init_optimizer(self, model, params):
        optimizer_type = params["type"]
        optimizer_args = params["args"]

        return getattr(optimizers_module, optimizer_type)(model.parameters(), **optimizer_args)

    def _init_criterion(self, params):
        criterion_type = params["type"]
        criterion_args = params["args"]

        return getattr(loss_module, criterion_type)(**criterion_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# Helper functions to update config dict with custom CLI options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


# Helper functions for reading/writing JSON
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
