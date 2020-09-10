# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import torch
import random

from torch import nn
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.modeling_roberta import RobertaModel
from torch.nn import Dropout

from .scalar_mix import ScalarMixWithDropout


class Wrapper(nn.Module):
    """Module for turning batches of sentences into BERT/RoBERTa/etc. embeddings."""
    def __init__(self, model_class, tokenizer_class, model_path, fine_tune=False, scalar_mix=False,
                 hidden_dropout=0.2, attn_dropout=0.2, output_dropout=0.5,
                 scalar_mix_layer_dropout=0.1, token_mask_prob=0.2):
        super(Wrapper, self).__init__()
        self.model = model_class.from_pretrained(model_path,
                                                 output_hidden_states=scalar_mix,
                                                 hidden_dropout_prob=hidden_dropout,
                                                 attention_probs_dropout_prob=attn_dropout)
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.token_mask_prob = token_mask_prob
        self.embedding_dim = self.model.config.hidden_size
        self.fine_tune = fine_tune

        if scalar_mix:
            num_layers = len(self.model.encoder.layer) + 1  # Add 1 because of input embeddings
            self.scalar_mix = ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=scalar_mix_layer_dropout)

        if output_dropout > 0.0:
            self.output_dropout = Dropout(p=output_dropout)

    def forward(self, input_sentences):
        """Transform a bunch of input sentences (list of lists of tokens) into a batch (tensor) of
        BERT/RoBERTa/etc. embeddings.
        """
        mask_prob = self.token_mask_prob if self.training else 0.0
        input_sequences = [BertInputSequence(sent, self.tokenizer, token_mask_prob=mask_prob) for sent in input_sentences]
        max_input_seq_len = max(len(input_sequence) for input_sequence in input_sequences)
        model_device = next(iter(self.model.parameters())).device
        for input_sequence in input_sequences:
            input_sequence.tensorize(model_device, padded_length=max_input_seq_len)

        # Batch components of input sequences
        input_ids = torch.stack([input_seq.token_ids for input_seq in input_sequences])
        attention_mask = torch.stack([input_seq.attention_mask for input_seq in input_sequences])
        original_token_mask = torch.stack([input_seq.original_token_mask for input_seq in input_sequences])

        batch_size = input_ids.shape[0]
        assert batch_size == len(input_sentences)

        # ---------------------------------------------------------------------------------------------------------
        # Step 1: Run tokens through model (with or without fine-tuning and/or scalar mixing)
        # ---------------------------------------------------------------------------------------------------------
        with torch.set_grad_enabled(self.fine_tune):
            if self.scalar_mix:
                output_layers = torch.stack(self.model(input_ids, attention_mask=attention_mask)[2])
                if self.output_dropout:
                    output_layers = self.output_dropout(output_layers)
                model_output = self.scalar_mix(output_layers)
            else:
                model_output = self.model(input_ids, attention_mask=attention_mask)[0]
                if self.output_dropout:
                    model_output = self.output_dropout(model_output)

        # ---------------------------------------------------------------------------------------------------------
        # Step 2: Reorder model output so that all embeddings corresponding to original tokens are at the beginning
        # ---------------------------------------------------------------------------------------------------------
        inverted_token_mask = 1 - original_token_mask

        # (The following three steps are needed because torch.argsort is not stable, i.e. we have to explicitly encode
        # the original order)
        multiplied_mask = inverted_token_mask * max_input_seq_len
        token_pos = torch.arange(0, max_input_seq_len, device=multiplied_mask.device).unsqueeze(0).expand((batch_size, max_input_seq_len))
        overall_mask = multiplied_mask + token_pos

        permutation = torch.argsort(overall_mask)
        output_reordered = torch.gather(model_output, 1, permutation.unsqueeze(-1).expand(model_output.shape))

        # ---------------------------------------------------------------------------------------------------------
        # Step 3: Cut off the excess embeddings so that the sequence length is reduced to the length of the longest
        # original (i.e. non-BERTified) sentence
        # ---------------------------------------------------------------------------------------------------------
        max_orig_seq_len = torch.max(torch.sum(original_token_mask, dim=1))
        output_stripped = output_reordered[:, 0:max_orig_seq_len, :]

        return output_stripped


class BertWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super(BertWrapper, self).__init__(BertModel, BertTokenizer, *args, **kwargs)


class RobertaWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super(RobertaWrapper, self).__init__(RobertaModel, RobertaTokenizer, *args, **kwargs)


class BertInputSequence:
    """Class for representing the features of a single, dependency-annotated sentence in tensor
       form, for usage in models based on BERT.

       Example (BERT):
           Input sentence:                Beware      the     jabberwock                 ,    my   son    !

           BERT tokens:          [CLS]    be ##ware   the     ja ##bber   ##wo  ##ck     ,    my   son    ! [SEP]  ([PAD] [PAD] [PAD]  ...)
           BERT token IDs:         101  2022   8059  1996  14855  29325  12155  3600  1010  2026  2365  999   102  (    0     0     0  ...)
           BERT attention mask:      1     1      1     1      1      1      1     1     1     1     1    1     1  (    0     0     0  ...)
           Original token mask:      0     1      0     1      1      0      0     0     1     1     1    1     0  (    0     0     0  ...)
    """

    def __init__(self, orig_tokens, bert_tokenizer, token_mask_prob=0.0):
        self.bert_tokenizer = bert_tokenizer

        self.tokens = [self.bert_tokenizer.cls_token]
        self.attention_mask = [1]
        self.original_token_mask = [0]  # [CLS] is not counted as an "original" token

        assert orig_tokens[0] != "[root]"  # Make sure legacy sentence representation is not used
        for orig_token in orig_tokens:
            if isinstance(self.bert_tokenizer, RobertaTokenizer):
                curr_bert_tokens = self.bert_tokenizer.tokenize(orig_token, add_prefix_space=True)
            else:
                curr_bert_tokens = self.bert_tokenizer.tokenize(orig_token)

            if token_mask_prob > 0.0 and random.random() < token_mask_prob:
                curr_bert_tokens = [self.bert_tokenizer.mask_token] * len(curr_bert_tokens)

            assert len(curr_bert_tokens) > 0

            self.tokens += curr_bert_tokens
            self.attention_mask += [1] * len(curr_bert_tokens)
            self.original_token_mask += [1] + [0] * (len(curr_bert_tokens) - 1)

        self.tokens.append(self.bert_tokenizer.sep_token)
        self.attention_mask.append(1)
        self.original_token_mask.append(0)

        assert len(orig_tokens) <= len(self.tokens) == len(self.attention_mask) == len(self.original_token_mask)

        # Convert BERT tokens to IDs
        self.token_ids = self.bert_tokenizer.convert_tokens_to_ids(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def pad_to_length(self, padded_length):
        """Pad the sentence to the specified length. This will increase the length of all fields to padded_length by
        adding the padding label/index.
        """
        assert padded_length >= len(self.tokens)
        padding_length = padded_length - len(self.tokens)

        self.tokens += [self.bert_tokenizer.pad_token] * padding_length
        self.token_ids += [0] * padding_length
        self.attention_mask += [0] * padding_length
        self.original_token_mask += [0] * padding_length

        assert len(self.tokens) == len(self.token_ids) == len(self.attention_mask) == len(self.original_token_mask)

    def tensorize(self, device, padded_length=None):
        """Convert the numerical fields of this BERT sentence into PyTorch tensors. The sentence may be padded to a
        specified length beforehand.
        """
        if padded_length is not None:
            self.pad_to_length(padded_length)

        self.token_ids = torch.tensor(self.token_ids, device=device)
        self.attention_mask = torch.tensor(self.attention_mask, device=device)
        self.original_token_mask = torch.tensor(self.original_token_mask, device=device)
