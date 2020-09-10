# This source code is from the PyTorch Template Project (w/ heavy adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import torch

from math import inf

from models.dependency_parser import DependencyParser


class Trainer:
    """Trainer class. Handles training logic as well as saving/loading model checkpoints."""

    def __init__(self, model, config, optimizer, criterion, train_data_loader, valid_data_loader):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # Set up GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Set up criterion (loss) and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # Set up epochs and checkpoint frequency
        cfg_trainer = config['trainer']
        self.min_epochs = cfg_trainer['min_epochs']
        self.max_epochs = cfg_trainer['max_epochs']
        self.save_period = cfg_trainer['save_period']
        self.start_epoch = 1
        self.early_stop = cfg_trainer.get('early_stop', inf)

        # Set up data loaders for training/validation examples
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        # Set up parser (used to compute validaction metrics, e.g. F-score)
        self.parser = DependencyParser(self.model)

        # Set up checkpoint saving and loading
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self):
        """Run the complete training logic as specified by the configuration."""
        not_improved_count = 0
        best_validation_fscore = 0.0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Perform one training epoch and output training metrics
            training_metrics = self.run_epoch(epoch, self.train_data_loader, training=True)
            self.logger.info("Training epoch {} finished.".format(epoch))
            self.log_metrics(training_metrics)

            # Perform one validation epoch and output validation metrics
            validation_metrics = self.run_epoch(epoch, self.valid_data_loader, training=False)
            self.logger.info("Validation epoch {} finished.".format(epoch))
            self.log_metrics(validation_metrics)

            # Check if model is new best according to validation F1 score
            improved = validation_metrics["fscore"] > best_validation_fscore
            if improved:
                best_validation_fscore = validation_metrics["fscore"]
                not_improved_count = 0
            else:
                not_improved_count += 1

            if improved or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=improved)

            if not_improved_count > self.early_stop and epoch >= self.min_epochs:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

    def run_epoch(self, epoch, data_loader, training=False):
        """Run one epoch on data provided by the given data loader.

        epoch: Integer, current epoch number.
        data_loader: Data loader to fetch examples from.
        training: If true, model will be trained (i.e. backpropagation takes place, dropout turned on).

        Returns: A dictionary that contains information about validation metrics (loss, precision, recall, f1).
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_metrics = {"loss": 0.0}
        overall_parsing_counts = {"correct": 0, "predicted": 0, "gold": 0}
        num_evaluated_batches = 0

        with torch.set_grad_enabled(training):
            for sentences, target in data_loader:
                # Run model
                target = self._to_device(target)
                output, parsing_counts = self.parser.evaluate_batch(sentences)

                # Compute loss
                output, target = self._unroll_sequence_batch(output), self._unroll_sequence_batch(target)
                loss = self.criterion(output, target)

                # Add metrics to overall total
                epoch_metrics["loss"] += loss.item()
                for count in "gold", "predicted", "correct":
                    overall_parsing_counts[count] += parsing_counts[count]

                # Perform backpropagation (when training)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Print progress
                num_evaluated_batches += 1
                self.logger.debug('{} Epoch: {} {} Loss: {:.6f}'.format(
                    "Training" if training else "Validation",
                    epoch,
                    self._progress(num_evaluated_batches, data_loader),
                    loss.item()))

        epoch_metrics.update(self.compute_prf(overall_parsing_counts))

        return epoch_metrics

    def compute_prf(self, counts_dict):
        """Compute precision, recall and F-score based on counts (gold, predicted, correct)."""
        precision = counts_dict["correct"] / counts_dict["predicted"] if counts_dict["predicted"] else 0.0
        recall = counts_dict["correct"] / counts_dict["gold"] if counts_dict["gold"] else 0.0
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {"precision": precision, "recall": recall, "fscore": fscore}

    def log_metrics(self, metrics):
        """Log evaluation metrics."""
        self.logger.info("Loss: {:.2f}".format(metrics["loss"]))
        self.logger.info("Precision: {:.2f}%".format(metrics["precision"] * 100))
        self.logger.info("Recall: {:.2f}%".format(metrics["recall"] * 100))
        self.logger.info("F-Score: {:.2f}%".format(metrics["fscore"] * 100))

    def _unroll_sequence_batch(self, batch):
        """Unroll a batch of sequences, i.e. flatten batch and sequence dimension. (Used for loss computation)"""
        shape = batch.shape
        if len(shape) == 3:  # Model output
            return batch.view(shape[0]*shape[1], shape[2])
        elif len(shape) == 2:  # Target labels
            return batch.view(shape[0]*shape[1])

    def _progress(self, num_completed_batches, data_loader):
        """Provide nicely formatted epoch progress."""
        return '[{}/{} ({:.0f}%)]'.format(num_completed_batches, len(data_loader),
                                          100.0 * num_completed_batches / len(data_loader))

    def _prepare_device(self, n_gpu_use):
        """Set up GPU device if available and move model to configured device."""
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint for the specified epoch number.

        If is_best is True, also save current checkpoint as 'model_best.pth'.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if is_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """Load model from a saved checkpoint."""
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['trainer']['optimizer']['type'] != self.config['trainer']['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch))

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            assert all(isinstance(val, torch.Tensor) for val in data.values())
            data_on_device = dict()
            for key in data:
                data_on_device[key] = data[key].to(self.device)
            return data_on_device
        else:
            raise Exception("Cannot move this kind of data to a device!")
