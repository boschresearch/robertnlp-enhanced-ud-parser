# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import re
import torch

from torch.nn.functional import softmax

from data_handling.dependency_sentence import DependencyAnnotatedSentence, heads, dependents


is_delexicalised_relation = lambda lbl: ":[" in lbl


class DependencyParser:
    """ The actual dependency parser. This can be thought of as a wrapper for the neural label scorer which transforms
    the outputted logits into dependency-annotated sentences. These can then be evaluated against a gold standard.
    """

    def __init__(self, label_scorer):
        self.label_scorer = label_scorer

    def parse(self, sentence):
        """Parse a singular sentence (in evaluation mode, i.e. no dropout) and perform post-processing.

        If sentence is of type str, input is assumed to be a whitespace-tokenized "raw" sentence. If sentence is of type
        DependencyAnnotatedSentence, tokenization will be taken from that sentence.

        Returns a DependencyAnnotatedSentence instance with the predicted relations.
        """
        # Extract sentence tokens and make dummy batch
        if isinstance(sentence, DependencyAnnotatedSentence):
            tokens = sentence.tokens[1:]
        elif isinstance(sentence, str):
            tokens = sentence.split(" ")
        else:
            raise Exception("Sentence must be either whitespace-tokenized string or DependencyAnnotatedSentence!")

        singleton_batch = [tokens]

        # Run label scorer (eval mode)
        self.label_scorer.eval()
        logits = self.label_scorer(singleton_batch)

        # Convert scorer output back into dependency labels
        logits = torch.squeeze(logits)
        labels = torch.argmax(logits, dim=1)

        # Make DependencyAnnotatedSentence and post-process
        parsed_sentence = DependencyAnnotatedSentence.from_tensor(["[root]"] + tokens, labels, self.label_scorer.output_vocab)
        self.post_process(parsed_sentence, logits)

        return parsed_sentence

    def evaluate_batch(self, annotated_sentences):
        """Run the parser on a batch of DependencyAnnotatedSentences and compute parsing metrics w.r.t. to the provided
        gold annotations. (No post-processing takes place.)

        Returns the raw model output as well as a dictionary containing the parsing metrics
        """
        logits = self.label_scorer([sent.tokens_no_root() for sent in annotated_sentences])
        labels = torch.argmax(logits, dim=2)

        assert len(labels) == len(annotated_sentences)  # Batch size must match

        batch_metrics = {"correct": 0, "predicted": 0, "gold": 0}
        for label_matrix, gold_sentence in zip(labels, annotated_sentences):
            predicted_sentence = DependencyAnnotatedSentence.from_tensor(gold_sentence.tokens, label_matrix,
                                                                         self.label_scorer.output_vocab)
            curr_metrics = DependencyAnnotatedSentence.get_parsing_counts(gold_sentence, predicted_sentence)
            for lbl in curr_metrics:
                for metric in "correct", "predicted", "gold":
                    batch_metrics[metric] += curr_metrics[lbl][metric]

        return logits, batch_metrics

    def post_process(self, raw_sentence, logits):
        """Post-process the dependencies a parsed sentence. Right now, this means
             (a) Attaching tokens which are not assigned a head
             (b) Removing superfluous heads of tokens which should only have one head (e.g. punctuation)
             (c) Adding lexical information to labels (e.g. obl:[case] -> obl:in)
        """
        for j in range(1, len(raw_sentence)):
            head_indices = list()
            head_relations = list()
            for i in range(len(raw_sentence)):
                head_relation = raw_sentence.dependencies[i][j]

                if head_relation != "[null]":
                    head_indices.append(i)
                    head_relations.append(head_relation)

                if is_delexicalised_relation(head_relation):
                    self.add_lexical_information(raw_sentence, i, j)

            if not head_relations:
                self.add_missing_head(raw_sentence, j, logits)
            elif self.inconsistent_heads(head_relations):
                self.remove_superfluous_heads(raw_sentence, head_indices, j, logits)

    def add_lexical_information(self, sentence, i, j):
        """Augment the delexicalised dependency relation i->j in this sentence by adding lexical information in a
        rule-based fashion.
        """
        head_relation = sentence.dependencies[i][j]
        stuff_before_delex, delex_relation, stuff_after_delex = head_relation.replace("]", "[").split("[")
        base_relation = stuff_before_delex.split(":")[0]

        # Step 1: Check if the token has a dependent that is attached via the delexicalised relation in question.
        # If so, lexicalise the relation with the form/lemma of that dependent
        for dep_ix, dep_rel in dependents(sentence, j):
            if dep_rel == delex_relation:
                lex_form = self.gather_lex(sentence, dep_ix)
                sentence.dependencies[i][j] = stuff_before_delex + lex_form + stuff_after_delex
                if sentence.dependencies[i][j].endswith(":"):
                    sentence.dependencies[i][j] = sentence.dependencies[i][j][:-1]
                return

        # Step 2: If step 1 did not work, do more advanced stuff

        # Step 2a: If base relation is not a conjunction and no direct dependent has been found, check if token has a
        # conjunction head. If yes, and if that head has a dependent with the delexicalised relation in question, pick
        # the form/lemma of that one.
        if base_relation != "conj":
            for head_ix, head_rel in heads(sentence, j):
                if head_rel.startswith("conj"):
                    for dep_ix, dep_rel in dependents(sentence, head_ix):
                        if dep_rel == delex_relation:
                            lex_form = self.gather_lex(sentence, dep_ix)
                            sentence.dependencies[i][j] = stuff_before_delex + lex_form + stuff_after_delex
                            if sentence.dependencies[i][j].endswith(":"):
                                sentence.dependencies[i][j] = sentence.dependencies[i][j][:-1]
                            return

        # Step 2b: If base relation is a conjunction, check if the token has a "conjunction sibling" with a dependent
        # of the delexicalised relation in question, and if so, pick that one. (This can happen in enumerations like
        # "x, y, and z")
        elif base_relation == "conj":
            for head_ix, head_rel in heads(sentence, j):
                if head_rel.startswith("conj"):
                    for conj_sibling_ix, conj_rel in dependents(sentence, head_ix):
                        if conj_rel.startswith("conj"):
                            for conj_sibl_dep_ix, conj_sibl_dep_rel in dependents(sentence, conj_sibling_ix):
                                if conj_sibl_dep_rel == delex_relation:
                                    lex_form = self.gather_lex(sentence, conj_sibl_dep_ix)
                                    sentence.dependencies[i][j] = stuff_before_delex + lex_form + stuff_after_delex
                                    if sentence.dependencies[i][j].endswith(":"):
                                        sentence.dependencies[i][j] = sentence.dependencies[i][j][:-1]
                                    return

        # Step 3: If neither step 1 nor step 2 worked, just return the non-lexicalised relation.
        sentence.dependencies[i][j] = (stuff_before_delex + stuff_after_delex).replace("::", ":")
        if sentence.dependencies[i][j].endswith(":"):
            sentence.dependencies[i][j] = sentence.dependencies[i][j][:-1]

    def add_missing_head(self, sentence, j, logits):
        """Attach the j'th token of the sentence via the second-highest-scoring relation."""
        # Extract the relevant logits for the j'th token and compute softmax
        relevant_logits = logits.view((len(sentence), len(sentence), -1))[:, j, :]
        probs = softmax(relevant_logits, dim=1)
        assert probs.shape == (len(sentence), len(self.label_scorer.output_vocab))

        # Best relation is the best non-null relation
        best = int(torch.argmax(probs[:, 1:]))

        # Extract head and relation type and write to sentence
        len_vocab_no_null = len(self.label_scorer.output_vocab) - 1
        head_ix = best // len_vocab_no_null
        relation_ix = (best % len_vocab_no_null) + 1

        sentence.dependencies[head_ix][j] = self.label_scorer.output_vocab.ix2token[relation_ix]

        # If added relation is delexicalised, lexicalise it
        if is_delexicalised_relation(sentence.dependencies[head_ix][j]):
            self.add_lexical_information(sentence, head_ix, j)

    def inconsistent_heads(self, head_relations):
        """Check if the given set of head relations is inconsistent (e.g. more than one punct relation)."""
        assert len(head_relations) >= 1

        if len(head_relations) == 1:
            return False

        # If the token has more than one head and is attached via one of the following relations,
        # something fishy is going on. (Note that this was determined empirically and may be language-dependent!)
        unitary_relations = {"fixed", "flat", "goeswith", "punct", "cc"}
        if set(head_relations) & unitary_relations:
            return True
        else:
            return False

    def remove_superfluous_heads(self, sentence, head_indices, j, logits):
        """Given a sentence and a token j, remove all head relations except the most confidently predicted one."""
        # Extract the relevant logits for the j'th token and its heads and compute softmax
        head_indices = torch.tensor(head_indices)
        relevant_logits = logits.view((len(sentence), len(sentence), -1))[head_indices, j, :]
        probs = softmax(relevant_logits, dim=1)
        assert probs.shape == (len(head_indices), len(self.label_scorer.output_vocab))

        # Best relation is the one with the highest top probability
        top_probs = torch.max(probs, dim=1)[0]
        best_top_prob_ix = torch.argmax(top_probs)
        best_head = head_indices[best_top_prob_ix]

        # Remove all head relations except the best one
        for i in range(len(sentence)):
            if i != best_head:
                sentence.dependencies[i][j] = "[null]"

    def gather_lex(self, sentence, token_ix):
        """Assemble multiword expressions for enhancement of relations (e.g. "as well as" -> "conj:as_well_as")"""
        mwe = sentence.tokens[token_ix].lower()
        for i in range(token_ix, len(sentence)):
            if sentence.dependencies[token_ix][i] == "fixed":
                mwe += "_{}".format(sentence.tokens[i].lower())

        mwe = re.sub("[^a-zA-Z]", "", mwe)  # Delete illegal characters to ensure validity

        return mwe

