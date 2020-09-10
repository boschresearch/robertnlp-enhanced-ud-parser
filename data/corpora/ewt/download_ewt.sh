#!/bin/bash

# Script for downloading the EWT corpus files as well as creating delexicalized versions of them.

# train
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -o en_ewt-ud-train.conllu
python ../delexicalize_corpus.py en_ewt-ud-train.conllu > en_ewt-ud-train.delex.conllu

# dev
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu -o en_ewt-ud-dev.conllu
python ../delexicalize_corpus.py en_ewt-ud-dev.conllu > en_ewt-ud-dev.delex.conllu


# test
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu -o en_ewt-ud-test.conllu
python ../delexicalize_corpus.py en_ewt-ud-test.conllu > en_ewt-ud-test.delex.conllu

