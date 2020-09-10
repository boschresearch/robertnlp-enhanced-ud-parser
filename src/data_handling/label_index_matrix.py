# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import itertools
import torch

from math import sqrt


class LabelIndexMatrix:
    """Class for representing a dependency matrix using the indices of dependency labels as provided by a vocabulary."""

    def __init__(self, size, padding_index=-1):
        """Initialize a DependencyLabelMatrix of a given size. The matrix will be completely filled with the
        provided padding label (default: -1).
        """
        self.padding_index = padding_index
        self.data = [[self.padding_index for i in range(size)] for j in range(size)]

    def __len__(self):
        if isinstance(self.data, torch.Tensor):
            return int(sqrt(len(self.data)))  # If data is a flat tensor, all rows of the
                                              # underlying matrix have been concatenated
        else:
            return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def pad_to_length(self, padded_length):
        """Pad the matrix to the specified length.

        This will increase the length of each row to padded_length (by appending self.padding_index) and increase the
        number of rows to padded_length by adding rows consisting only of self.padding_index. I.e., after this operation
        the dependency matrix has size padded_length*padded_length.
        """
        assert padded_length >= len(self)
        padding_length = padded_length - len(self)

        for row in self:
            row += [self.padding_index] * padding_length

        for _ in range(padding_length):
            self.data.append([self.padding_index] * padded_length)

    def tensorize(self, padded_length=None):
        """Convert this matrix to a "flat" PyTorch tensor containing the data of the dependency matrix (optionally
        padding to a specified length beforehand).

        The resulting tensor is 1-dimensional and contains the concatenation of all the rows of the matrix.
        """
        if padded_length is not None:
            self.pad_to_length(padded_length)

        deps_flat = list(itertools.chain(*self.data))

        self.data = torch.tensor(deps_flat)

    @staticmethod
    def from_label_matrix(dependency_matrix, label_vocab, padding_index=-1):
        """Create a DependencyLabelMatrix from a matrix (list of lists) of dependency label strings."""
        matrix = LabelIndexMatrix(len(dependency_matrix), padding_index=padding_index)

        for i in range(len(dependency_matrix)):
            for j in range(len(dependency_matrix)):
                matrix.data[i][j] = label_vocab.token2ix[dependency_matrix[i][j]]

        return matrix

    @staticmethod
    def from_tensor(dep_tensor, sent_length):
        """Convert a "flat" label tensor into a LabelIndexMatrix."""
        assert len(dep_tensor.shape) == 1
        assert len(dep_tensor) >= sent_length**2  # Dependencies tensor must be large enough to contain
                                                  # relations between all word pairs
        matrix = LabelIndexMatrix(sent_length)

        tensor_size = int(sqrt(len(dep_tensor)))
        assert tensor_size**2 == len(dep_tensor)  # Tensor must encode a square matrix

        for i in range(sent_length):
            for j in range(sent_length):
                matrix.data[i][j] = int(dep_tensor[i*tensor_size + j])

        return matrix
