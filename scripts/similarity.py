import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

import argparse

def similarity(df_pars:pd.DataFrame, df_sents:pd.DataFRame, queries, model_name):
    """
    
    """
    print("Searching and computing embeddings for relevant sentences ...")
    rel_sents = []
    for queiry in queries:
        rel_sents =  rel_sents + df_sents[df_sents['text'].str.contains(queiry)]['text'].to_list()
    rel_sents = list(set(rel_sents))
    
    print(f"N of matched sentences: {len(rel_sents)}")

    model = SentenceTransformer(model_name)

    print("Matched sentences embedding computing ...")
    rel_embeddings = model.encode(rel_sents)
    rel_mean_embedding = rel_embeddings.mean(axis = 0)

    print("Paragraphs embeddings computing ...")
    par_embeddings = model.encode(df_pars['text'])
    print("Paragraphs similarity computing ...")
    df_pars['cos_sim'] = [cosine_similarity(par_embedding.reshape(1, -1), rel_mean_embedding.reshape(1, -1))[0][0] for par_embedding in tqdm(par_embeddings)]

    print("Sentences embeddings computing ...")
    sent_embeddings = model.encode(df_sents['text'])
    print("Sentences similarity computing ...")
    df_sents['cos_sim'] = [cosine_similarity(sent_embedding.reshape(1, -1), rel_mean_embedding.reshape(1, -1))[0][0] for sent_embedding in tqdm(sent_embeddings)]

    return df_pars, df_sents, par_embeddings, sent_embeddings

def run(args):
    """
    
    """
    queries = []
    with open(args.queries, "r") as f:
        queries.append(f.readline())
    
    df_pars = pd.read_csv(args.input + "paragraphs.csv")
    df_sents = pd.read_csv(args.input + "sentences.csv")

    df_pars_sim, df_sents_sim, par_embeddings, sent_embeddings= similarity(df_pars, df_sents, queries, args.model)
    print("Saving paragraphs embeddings ...")
    np.save(args.output+"paragraphs_embeddings.npy", par_embeddings)
    print("Saving paragraphs with similarity score ...")
    df_pars_sim.to_csv(args.output + "paragraphs_sim.csv", index = False)
    print("Saving sentences embeddings ...")
    np.save(args.output + "sentences_embeddings.npy", sent_embeddings)
    print("Saving sentences with similarity score ...")
    df_sents_sim.to_csv(args.output + "sentences_sim.csv", index = False)

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("queries", help = "path to file with regex queries for relevant sentences search", type = str)
    parser.add_argument("-i", "--input", help = "path to directory with paragraphs.csv and sentences.csv (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-o", "--output", help = "path to directory where files will be stored (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-e", "--embeddings", help = "is there embeddings ")
    parser.add_argument("-m", "--model", help = "model for embedding (default: sentence-transformers/sentence-t5-xl)", type = str, default = "sentence-transformers/sentence-t5-xl")

    args = parser.parse_args()
    run(args)