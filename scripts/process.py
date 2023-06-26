import split
import similarity
import topic_modeling

import argparse

def run(args):
    split.run(args)
    args.input = args.output
    similarity.run(args)
    topic_modeling.run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help = "path to input file", type = str)
    parser.add_argument("queries", help = "path to file with regex queries for relevant sentences search", type = str)
    parser.add_argument("-o", "--output", help = "path to directory where output files will be stored (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-l", "--lang", help = "language of documents (default: en)", type = str, default = "en")
    parser.add_argument("-s", "--smart", help = "use smart paragraphisation", action="store_true")
    parser.add_argument("-m", "--model", help = "model for embedding (default: sentence-transformers/sentence-t5-xl)", type = str, default = "sentence-transformers/sentence-t5-xl")
    parser.add_argument("-t", "--threshold", help = "threshold to determine relevant sentences (default: 0.5)", type = float, default = 0.5)
    parser.add_argument("-sm", "--spacy_model", help = "spacy model for lemmatization (default: en_core_web_lg)", type = str, default = "en_core_web_lg")
    parser.add_argument("-gpt", "--gpt_model", help = "model for topic representation and summary (default: None)", type = str, default = None)

    args = parser.parse_args()
    run(args)