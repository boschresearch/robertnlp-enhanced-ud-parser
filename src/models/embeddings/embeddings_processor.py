# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import torch

from torch import nn
from torch.nn import ModuleList


class EmbeddingsProcessor(nn.Module):
    """Module for turning sentences into sequences of token embeddings.

    The module takes as input a list of embedding wrappers (e.g. BertWrapper), which handle the nitty-gritty of
    turning tokens into vector representations. The outputs of these wrappers are then concatenated to form the final
    embedding sequence for the sentence.
    """

    def __init__(self, input_embeddings):
        super(EmbeddingsProcessor, self).__init__()

        self.input_embeddings = ModuleList(input_embeddings)
        self.output_dim = sum(input_embedding.embedding_dim for input_embedding in input_embeddings)

        self.root_embedding = nn.Parameter(torch.randn(self.output_dim), requires_grad=True)

    def forward(self, input_sentence_batch):
        """Maps the input sentence bach (list of lists of tokens) to a tensor of token embeddings. Output is padded to
        maximum sequence length and thus of shape (batch_size, max_seq_length, output_dim).

        Note that we are assuming here that the input sentences do *not* have an artificial [root] token as their
        first token.

        Returns the sentence batch tensor as well as the padding mask (BoolTensor in which True values represent padding
        at this index.)
        """
        # Run sentences through all input embeddings and concatenate them in one tensor
        embedded_sentences = [input_embedding(input_sentence_batch) for input_embedding in self.input_embeddings]
        sentence_batch_tensor = torch.cat(embedded_sentences, dim=2)

        # Insert learned [root] embedding at beginning of each sentence
        batch_size = len(sentence_batch_tensor)
        root_embedding_expanded = self.root_embedding.unsqueeze(0).unsqueeze(0).expand((batch_size, 1, self.output_dim))
        sentence_batch_tensor = torch.cat((root_embedding_expanded, sentence_batch_tensor), dim=1)

        return sentence_batch_tensor, padding_mask(input_sentence_batch)


def padding_mask(input_sentence_batch):
    """Returns a BoolTensor indicating which positions will be padding when turning this batch of sentences
    into a tensor."""
    max_sent_length = max(len(sent) + 1 for sent in input_sentence_batch)  # Add 1 because of [root] token
    padding_masks = list()
    for sentence in input_sentence_batch:
        sent_length = len(sentence) + 1  # Again add 1 because of [root] token
        padding_mask = [0 for _ in range(sent_length)] + [1 for _ in range(max_sent_length - sent_length)]
        padding_masks.append(padding_mask)
    return torch.cuda.BoolTensor(padding_masks)
