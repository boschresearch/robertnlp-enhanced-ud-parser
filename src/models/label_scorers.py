# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

from torch import nn


class BasicLabelScorer(nn.Module):
    """Module for classifying dependency relations between tokens (represented as token embeddings)."""

    def __init__(self, embeddings_processor, classifier, output_vocab):
        super(BasicLabelScorer, self).__init__()

        # Source of token embeddings
        self.embed = embeddings_processor

        # Classifier that computes the actual logits for all token pairs
        self.classifier = classifier

        # Output vocabulary
        self.output_vocab = output_vocab

    def forward(self, input_sentences):
        # Get token embeddings
        embeddings, padding_mask = self.embed(input_sentences)
        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]

        # Run the classifier on all the token pairs
        logits = self.classifier(embeddings, embeddings)

        # Return "flattened" version of the logits
        return logits.view(batch_size, seq_len*seq_len, -1)
