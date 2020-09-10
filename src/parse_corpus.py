# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import argparse

from init_config import ConfigParser
from data_handling.conllu_dataset import CoNLLUDataset


def main(config, corpus_filename, conllu=False):
    """Parse each sentence of the input corpus and write to stdout.

    If conllu==False, assumes whitespace tokenization and one sentence per line.
    If conllu==True, read sentences from a corpus file in CoNLL-U format instead.
    """
    model = config.init_model()

    trainer = config.init_trainer(model, None, None)  # Somewhat inelegant, but need to do this because
                                                      # trainer handles checkpoint loading
    parser = trainer.parser

    if conllu:  # Parse from CoNLL-U file
        dataset = CoNLLUDataset.from_corpus_file(corpus_filename)
        for sentence in dataset:
            parsed_sentence = parser.parse(sentence)
            print(parsed_sentence.to_conllu())
    else:  # Parse from tokenized text
        with open(corpus_filename, "r") as corpus_file:
            for line in corpus_file:
                parsed_sentence = parser.parse(line.strip())
                print(parsed_sentence.to_conllu())


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Graph-based enhanced UD parser (corpus parsing mode)')
    args.add_argument('-c', '--conllu', action='store_true', help='load corpus from CoNLL-U file')
    args.add_argument('config', type=str, help='config file path (required)')
    args.add_argument('resume', type=str, help='path to model checkpoint to be loaded (required)')
    args.add_argument('corpus', type=str, help='path to corpus file (required)')

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args.corpus, args.conllu)
