import sys
import stanfordnlp
import code

from stanfordnlp import Pipeline

if __name__ == "__main__":
    pipeline = stanfordnlp.Pipeline(processors='tokenize', treebank='en_ewt')
    doc = pipeline(open(sys.argv[1]).read())
    doc.write_conll_to_file("tokenizer_output.conllu")

