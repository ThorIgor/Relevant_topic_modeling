import numpy as np
import pandas as pd

import spacy
import re

from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

import argparse

def rev_sigmoid(x:float)->float:
    """
    Reverse sigmoid function.
    
    Args:
        x (float): Input value.
    
    Returns:
        float: Reverse sigmoid value.
    """
    return (1 / (1 + np.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
    """
    Calculate weighted sums of activated sentence similarities.
    
    Args:
        similarities (numpy array): Square matrix where each sentence corresponds to another with cosine similarity.
        p_size (int): Number of sentences used to calculate weighted sum.
    
    Returns:
        numpy array: List of weighted sums.
    """
    # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10,10,p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
    ### 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
    ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
    ### 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    ### 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1,1)
    ### 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities

def ordinary_paragraphs(df:pd.DataFrame)->pd.DataFrame:
    """
    Split text into ordinary paragraphs.
    
    Args:
        df (pandas DataFrame): DataFrame with 'fulltext' column containing the text.
    
    Returns:
        pandas DataFrame: DataFrame with paragraphs.
    """
    df_pars = []
    id = 0
    print("Ordinary paragraphs ...")
    for i in tqdm(df.index):
        sp = re.split("([.?!])(?:\n)", df['fulltext'][i])
        pars = []
        for s in sp:
            if s in [".", '?', "!", ""]:
                pars[-1] += s
            else:
                pars.append(s)
        for par in pars:
            df_pars.append({
                "doc_id": df['id'][i],
                "id": id,
                "text": par
        })
        id += 1
    df_pars = pd.DataFrame(df_pars)
    return df_pars

def smart_paragraphs(df:pd.DataFrame, model_name:str = "sentence-transformers/sentence-t5-xl", lang:str = "en")->pd.DataFrame:
    """
    Split text into smart paragraphs using sentence embeddings.
    
    Args:
        df (pandas DataFrame): DataFrame with 'fulltext' column containing the text.
        model_name (str): Name or path of the sentence embedding model to use (default: "sentence-transformers/sentence-t5-xl").
        lang (str): Language of the documents (default: "en").
    
    Returns:
        pandas DataFrame: DataFrame with smart paragraphs.
    """
    print("Loading embedding model ...")
    model = SentenceTransformer(model_name)

    nlp = spacy.blank(lang)
    nlp.add_pipe('sentencizer')

    df_smart_pars = []
    id = 0
    print("Smart paragrpahs ...")
    for doc_i in tqdm(df.index):
        par_list = []

        sents = [sent.text.replace("\n", " ") for sent in nlp(df['fulltext'][doc_i]).sents]
        embeddings = model.encode(sents)
        sims = cosine_similarity(embeddings)

        act_sims = activate_similarities(sims, p_size = np.min([len(sents), 10]))
        loc_min_i = argrelextrema(act_sims, np.less, order = 2)[0]

        last_i = 0
        for i in loc_min_i:
            par_list.append(" ".join(sents[last_i:i]))
            last_i = i
        par_list.append(" ".join(sents[last_i:len(sents)]))
        for par in par_list:
            df_smart_pars.append({
                "doc_id": df['id'][doc_i],
                "id": id,
                "text": par
            })
            id+=1

    df_smart_pars = pd.DataFrame(df_smart_pars)
    return df_smart_pars

def sentences(df_pars:pd.DataFrame, lang:str = "en")->pd.DataFrame:
    """
    Split paragraphs into sentences.
    
    Args:
        df_pars (pandas DataFrame): DataFrame with paragraphs.
        lang (str): Language of the documents (default: "en").
    
    Returns:
        pandas DataFrame: DataFrame with sentences.
    """
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    df_sents = []
    id = 0
    print("Sentences ...")
    for i in tqdm(df_pars.index):
        for sent in nlp(df_pars['text'][i]).sents:
            df_sents.append({
                "doc_id": df_pars['doc_id'][i],
                "par_id": df_pars['id'][i],
                "id": id,
                "text": sent.text
            })
            id+=1
    df_sents = pd.DataFrame(df_sents)
    return df_sents

def run(args):
    """
    Run paragraphs and sentences split process.
    
    Args:
        args: Command-line arguments passed to the script.
    """
    extension = args.input.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(args.input)
    elif extension == "json":
        df = pd.read_json(args.input)
    else:
        raise Exception("Not supported extension")

    df = df[(df['fulltext'] != '') & ~df['fulltext'].isna()]
    df.reset_index(inplace = True)
    df.drop("index", axis = 1, inplace = True)
    df['id'] = df.index
    
    if args.smart:
        df_pars = smart_paragraphs(df, args.model, args.lang)
    else:
        df_pars = ordinary_paragraphs(df)

    print(f"N of paragraphs: {df_pars.shape[0]}")

    df_sents = sentences(df_pars, args.lang)

    print(f"N of sentences: {df_sents.shape[0]}")

    print("Saving documents ...")
    df.to_csv(args.output + "documents.csv", index = False)
    print("Saving paragraphs ...")
    df_pars.to_csv(args.output + "paragraphs.csv", index = False)
    print("Saving sentences ...")
    df_sents.to_csv(args.output + "sentences.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help = "path to input file", type = str)
    parser.add_argument("-o", "--output", help = "path to directory where output files will be stored (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-l", "--lang", help = "language of documents (default: en)", type = str, default = "en")
    parser.add_argument("-s", "--smart", help = "use smart paragraphisation", action="store_true")
    parser.add_argument("-m", "--model", help = "model for smart paragraphisation (default: sentence-transformers/sentence-t5-xl)", type = str, default = "sentence-transformers/sentence-t5-xl")

    args = parser.parse_args()
    run(args)