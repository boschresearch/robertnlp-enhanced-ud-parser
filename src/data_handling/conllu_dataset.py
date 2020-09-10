# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import pyconll

from torch.utils.data import Dataset

from .vocab import BasicVocab
from .dependency_sentence import DependencyAnnotatedSentence


class CoNLLUDataset(Dataset):
    """Class for representing a dataset of CoNLL-U sentences.

    The individual objects contained within are of type CoNLLUSentence.
    """

    def __init__(self):
        self.sentences = list()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def append_sentence(self, sent):
        """Add one sentence to the dataset."""
        self.sentences.append(sent)

    @staticmethod
    def from_corpus_file(corpus_filename):
        """Read in a dataset from a corpus file in CoNLL-U format."""
        dataset = CoNLLUDataset()

        raw_sents = pyconll.load_from_file(corpus_filename)
        for raw_sent in raw_sents:
            processed_sent = DependencyAnnotatedSentence.from_conllu(raw_sent)
            dataset.append_sentence(processed_sent)

        return dataset

    @staticmethod
    def extract_dep_label_vocab(*conllu_datasets, null_label="[null]"):
        """Extract a vocabulary of dependency labels from one or more CONLL-U datasets."""
        vocab = BasicVocab()

        if null_label is not None:
            vocab.add(null_label)

        for dataset in conllu_datasets:
            for sentence in dataset:
                for dep_label in [lbl for head_row in sentence.dependencies for lbl in head_row]:
                    vocab.add(dep_label)

        assert vocab.is_consistent()
        return vocab
