#!/bin/bash

# Locations of UDify components
UDIFY_PATH=~/Documents/parsing/udify/  # Replace this with the the location of your UDify directory
UDIFY_ARCHIVE_PATH=~/Documents/parsing/udify/udify-model.tar.gz  # Replace this with the the location of your downloaded UDify model archive (udify-model.tar.gz)

# Tokenize and segment the input text
python src/pre_post_processing/tokenize_input.py $3

# Parse text
python src/parse_corpus.py --conllu $1 $2 tokenizer_output.conllu > parsed_raw.conllu

# Post-process parser output with UDify
python src/pre_post_processing/repair_graphs.py $UDIFY_PATH $UDIFY_ARCHIVE_PATH parsed_raw.conllu > parsed.conllu

# Delete temp files
rm tokenizer_output.conllu
rm parsed_raw.conllu

