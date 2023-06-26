import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import numpy as np
import pandas as pd

import spacy

from bertopic import BERTopic
from keybert import KeyBERT
import plotly.io as pio

import openai
import tiktoken
from bertopic.representation import OpenAI

from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import argparse
import os
import gc

tqdm.pandas()

class LemmaTokenizer:
    def __init__(self, nlp_str):
        self.nlp = spacy.load(nlp_str)
    def __call__(self, doc):
      def chunkstring(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))
      if len(doc) > 1000000:
        lemms = []
        for chunk in chunkstring(doc, 500000):
          lemms = lemms + self.__call__(chunk)
        return lemms
      else:
        return [t.lemma_ for t in self.nlp(doc) if not t.is_punct]

def extract_keyNPs(df:pd.DataFrame, embedding_model:str, spacy_model:str, embeddings:np.array = None)->pd.DataFrame:
  """
    Extract key noun phrases (keyNPs) using KeyBERT.

    Args:
        df (pd.DataFrame): DataFrame with sentences.
        embedding_model (str): Name of the SentenceTransformer model for embeddings.
        spacy_model (str): Name of the spacy model for lemmatization.
        embeddings (np.array, optional): Array containing sentence embeddings. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with extracted keyNPs and lemmatized keyNPs.
  """
  if not spacy.util.is_package(spacy_model):
     print("Downloading spacy model ...")
     spacy.cli.download(spacy_model)

  vectorizer = KeyphraseCountVectorizer(spacy_pipeline=spacy.load(spacy_model), pos_pattern='<ADJ.*>*<N.*>+')
  keybert = KeyBERT(model = SentenceTransformer(embedding_model))

  print("Extracting keyNPs with keyBERT ...")
  keynps = []
  for i in tqdm(range(df.shape[0])):
     keynps.append(keybert.extract_keywords(df['text'].iloc[i], vectorizer = vectorizer, doc_embeddings = embeddings[i, :].reshape(1, -1)))
  df['keyNPs'] = keynps

  print("Lemmatization of keyNPs ...")
  nlp = spacy.load(spacy_model)
  df['keyNPs_lemm'] = df['keyNPs'].progress_apply(lambda x: [' '.join([t.lemma_ for t in nlp(np[0]) if not t.is_punct]) for np in x])

  return df

def topic_modeling(df:pd.DataFrame, embedding_model:str, spacy_model:str, embeddings:np.array = None)->BERTopic:
    """
      Fit topic model (BERTopic)

      Args:
        df (pd.DataFrame): DataFrame with sentences and keyNPs
        embeddings_model (str): Name of the SentenceTransformer model for embeddings
        spacy_model (str): Name of the spacy model for lemmatization
        embeddings (np.array, optinal): Array containing sentence embeddings. Defaults to None

      Reurns:
        model (bertopic.BERTopic): topic model
    """
    if not spacy.util.is_package(spacy_model):
       print("Downloading spacy model ...")
       spacy.cli.download(spacy_model)

    vocabulary = []
    for keynps in df['keyNPs_lemm']:
        vocabulary = vocabulary + keynps
    vocabulary = list(set(vocabulary))

    stopwords = list(spacy.load(spacy_model).Defaults.stop_words)
    vectorizer_model = CountVectorizer(tokenizer=LemmaTokenizer(spacy_model), ngram_range=(1,3), stop_words = stopwords, vocabulary = vocabulary)
    model = BERTopic(embedding_model = SentenceTransformer(embedding_model), nr_topics = "auto", vectorizer_model=vectorizer_model, verbose = True)
   
    print("Fitting BERTopic model ...")
    _ = model.fit_transform(df['text'], embeddings = embeddings.reshape(df.shape[0], -1))

    return model

def topic_representation(api_key, gpt_model, topic_model, df):
    """
      Get new topic representation and summary

      Args:
        api_key (str): openAI API key
        gpt_model (str): model to use
        topic_model (bertopic.BERTopic): topic model
        df (pd.DataFrame): DataFrame with sentences

      Returns:
        new_topics (dict): dict with new topic representations
        summary (dict): dict with summary for each topic
    """
    
    print("Topic representation ...")
    openai.api_key = api_key

    encoding = tiktoken.encoding_for_model(gpt_model)
    mean_len = df['text'].apply(lambda x: len(encoding.encode(x))).mean()

    gpt_model = OpenAI(model=gpt_model, delay_in_seconds=60, chat=True, nr_docs = min(int(4096/mean_len), 50))

    documents = topic_model.get_document_info(df['text'])
    documents['Representation'] = documents['Representation'].apply(lambda x: tuple(x))
    documents['Representative_Docs'] = documents['Representative_Docs'].apply(lambda x: tuple(x))

    try:
      new_topics = gpt_model.extract_topics(topic_model, documents, topic_model.c_tf_idf_, topic_model.get_topics())
    except openai.error.APIError as ex:
      print("!!! OpenAI APIError !!!")
      print(ex)

    gpt_model.prompt = """
      I have a topic that is described by the following keywords: [KEYWORDS]
      In this topic, the following documents are a small but representative subset of all documents in the topic:
      [DOCUMENTS]

      Based on the information above, please give a description of this topic in the following format:
      topic: <description>
      """
    
    print("Summary ...")
    try:
      summary = gpt_model.extract_topics(topic_model, documents, topic_model.c_tf_idf_, topic_model.get_topics())
    except openai.error.APIError as ex:
      print("!!! OpenAI APIError !!!")
      print(ex)
       

    return new_topics, summary

def run(args):
    """
      Run the topic modeling

      Args:
        args: Command-line arguments passed to the script.
    """

    embeddings = None
    if "sentences_embeddings.npy" in os.listdir(args.input):
       embeddings = np.load(args.input + "sentences_embeddings.npy")
    
    datetime = None
    if "documents.csv" in os.listdir(args.input):
      docs = pd.read_csv(args.input + "documents.csv")
      if 'datetime' in docs.columns:
         datetime = docs[['id', 'datetime']]
      del docs
      gc.collect()

    df = pd.read_csv(args.input + "sentences_sim.csv")
    df = df[df['cos_sim'] > args.threshold]
    if datetime is not None:
      df = pd.merge(df, datetime, "left", left_on = "doc_id", right_on = "id", left_index=True)
    if embeddings is not None:
      embeddings = embeddings[df.index, :]
    df.reset_index(inplace = True)
    df.drop("index", axis = 1, inplace = True)

    print(f"N sentences over threshold: {df.shape[0]}")

    df = extract_keyNPs(df, args.model, args.spacy_model, embeddings)

    print("Saving sentences with keyNPs ...")
    df.to_csv(args.output + "sentences_keyNPs.csv", index = False)

    model = topic_modeling(df, args.model, args.spacy_model, embeddings)

    print("Saving raw topics ...")
    with open(args.output + "topics_raw.txt", "w+") as f:
      f.write('\n'.join([str(topic) + ". " + ', '.join([t[0] for t in model.get_topics()[topic]]) for topic in model.get_topics()]))
    

    if args.gpt_model is not None:
      if args.api_key is None:
         raise RuntimeError("No gpt key provided")
      new_topics, summary = topic_representation(args.api_key, args.gpt_model, model, df)
    else:
      new_topics = model.generate_topic_labels(nr_words=7, topic_prefix=True, separator=", ")
      summary = None
    model.set_topic_labels(new_topics)

    print("Saving updated topics ...")
    with open(args.output + "topics_updated.txt", "w+") as f:
      f.write('\n'.join([str(key) + ". " + new_topics[key][0][0] for key in new_topics]))
    if summary is not None:
      with open(args.output + "summary.txt", "w+") as f:
         f.write('\n'.join([str(key) + ". " + summary[key][0][0] for key in summary]))

    print("Saving visuals ...")
    lens = []
    for topic in model.get_topics().values():
      lens.append(len(" ".join([word for word, _ in topic])) + 3)
    width = max(lens)*3 + 500

    try:
      model.visualize_topics(custom_labels = True, width = width).write_html(args.output + "topics_vis.html")
    except ValueError:
      print("Imposible to create topics_vis")
    
    model.visualize_documents(df['text'], custom_labels = True).write_html(args.output + "documents_vis.html")

    if 'datetime' in df.columns:
      df['datetime'] = pd.to_datetime(df['datetime'], yearfirst = True)
      df['year_month'] = df['datetime'].apply(lambda x: str(x.year) + "-" + str(x.month))
      topics_over_time = model.topics_over_time(df['text'], df['year_month'], evolution_tuning=False, global_tuning=False)
      model.visualize_topics_over_time(topics_over_time, custom_labels = True, width = width).write_html(args.output + "over_time_vis.html")
    else:
       print("Imposible to create over_time_vis")

    model.visualize_barchart(custom_labels = True, n_words = 10, width = width).write_html(args.output + "barchar_vis.html")

    hierarchical_topics = model.hierarchical_topics(df['text'])
    model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels = True, width = width).write_html(args.output + "hierarchy_vis.html")

    model.visualize_heatmap(custom_labels = True, width = width).write_html(args.output + "heatmap_vis.html")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help = "path to directory with sentences_sim.csv, optionaly with sentences_embeddings.npy, documents.csv (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-o", "--output", help = "path to directory where files will be stored (default: ../data/)", type = str, default = "../data/")
    parser.add_argument("-t", "--threshold", help = "threshold to determine relevant sentences (default: 0.5)", type = float, default = 0.5)
    parser.add_argument("-sm", "--spacy_model", help = "spacy model for lemmatization (default: en_core_web_lg)", type = str, default = "en_core_web_lg")
    parser.add_argument("-m", "--model", help = "model for embedding (default: sentence-transformers/sentence-t5-xl)", type = str, default = "sentence-transformers/sentence-t5-xl")
    parser.add_argument("-gpt", "--gpt_model", help = "model for topic representation and summary (default: None)", type = str, default = None)
    parser.add_argument("-ak", "--api_key", help = "openAI API key (default: None)", type = str, default = None)

    args = parser.parse_args()
    run(args)