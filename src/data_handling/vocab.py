# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald


class BasicVocab:
    """Class for mapping labels/tokens to indices and vice versa."""

    def __init__(self, vocab_filename=None):
        """Read a vocabulary from a file."""
        self.ix2token = list()
        self.token2ix = dict()

        if vocab_filename is not None:
            with open(vocab_filename) as vocab_file:
                for ix, line in enumerate(vocab_file):
                    token = line.strip()

                    self.ix2token.append(token)
                    self.token2ix[token] = ix

        assert self.is_consistent()

    def __len__(self):
        return len(self.ix2token)

    def __str__(self):
        return "\n".join(self.ix2token)

    def add(self, token):
        """Add a token to the vocabulary if it does not already exist."""
        if token not in self.token2ix:
            new_ix = len(self.ix2token)

            self.token2ix[token] = new_ix
            self.ix2token.append(token)

    def to_file(self, vocab_filename):
        """Write vocabulary to a file."""
        with open(vocab_filename, "w") as vocab_file:
            for token in self.ix2token:
                vocab_file.write(token + "\n")

    def is_consistent(self):
        """Checks if all index mappings match up. Used for debugging."""
        if len(self.ix2token) != len(self.token2ix):
            return False

        try:
            for token, ix in self.token2ix.items():
                if self.ix2token[ix] != token:
                    return False
        except IndexError:
            return False

        return True
