# Relevant topic modeling

This is repository with scripts for similarity search and topic modeling

# Preperation

<details>
<summary>Creating and activating virtual environment (optional)</summary>
Creating virtual environment

```
python -m venv venv
```
Activating virtual environment

Windows

```
venv\Scripts\activate.bat
```

Linux

```
source <venv>/bin/activate
```

</details>

Requirements installation

```
pip install -r requirenments.txt
```

# Full process

Example:

```
python scripts/process.py example/test.csv example/queries.txt -o data/
```

<details>
<summary>process.py usage</summary>
  
``` 
usage: process.py [-h] [-o OUTPUT] [-l LANG] [-s] [-m MODEL] [-t THRESHOLD] [-sm SPACY_MODEL] [-gpt GPT_MODEL]
                  input queries

positional arguments:
  input                 path to input file
  queries               path to file with regex queries for relevant sentences search

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to directory where output files will be stored (default: ../data/)
  -l LANG, --lang LANG  language of documents (default: en)
  -s, --smart           use smart paragraphisation
  -m MODEL, --model MODEL
                        model for embedding (default: sentence-transformers/sentence-t5-xl)
  -t THRESHOLD, --threshold THRESHOLD
                        threshold to determine relevant sentences (default: 0.5)
  -sm SPACY_MODEL, --spacy_model SPACY_MODEL
                        spacy model for lemmatization (default: en_core_web_lg)
  -gpt GPT_MODEL, --gpt_model GPT_MODEL
                        model for topic representation and summary (default: None)
```

</details>

# Paragraphs and sentences split process

Example:

```
python scripts/split.py example/test.csv -o data/
```

<details>
<summary>split.py usage</summary>
  
```
usage: split.py [-h] [-o OUTPUT] [-l LANG] [-s] [-m MODEL] input

positional arguments:
  input                 path to input file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to directory where output files will be stored (default: ../data/)
  -l LANG, --lang LANG  language of documents (default: en)
  -s, --smart           use smart paragraphisation
  -m MODEL, --model MODEL
                        model for smart paragraphisation (default: sentence-transformers/sentence-t5-xl)
```

</details>

# Similarity score computing

Example:

```
python scripts/similarity.py example/queries.txt -i data/ -o data/
```

<details>
<summary>similarity.py usage</summary>
  
```
usage: similarity.py [-h] [-i INPUT] [-o OUTPUT] [-e EMBEDDINGS] [-m MODEL] queries

positional arguments:
  queries               path to file with regex queries for relevant sentences search

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to directory with paragraphs.csv and sentences.csv (default: ../data/)
  -o OUTPUT, --output OUTPUT
                        path to directory where files will be stored (default: ../data/)
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        is there embeddings
  -m MODEL, --model MODEL
                        model for embedding (default: sentence-transformers/sentence-t5-xl)
```

</details>

# Topic modeling

Example:

```
python scripts/topic_modeling.py -i data/ -o data/
```

<details>
<summary>topic_modeling.py usage</summary>
  
```
usage: topic_modeling.py [-h] [-i INPUT] [-o OUTPUT] [-t THRESHOLD] [-sm SPACY_MODEL] [-m MODEL] [-gpt GPT_MODEL]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to directory with sentences_sim.csv, optionaly with sentences_embeddings.npy, documents.csv (default:
                        ../data/)
  -o OUTPUT, --output OUTPUT
                        path to directory where files will be stored (default: ../data/)
  -t THRESHOLD, --threshold THRESHOLD
                        threshold to determine relevant sentences (default: 0.5)
  -sm SPACY_MODEL, --spacy_model SPACY_MODEL
                        spacy model for lemmatization (default: en_core_web_lg)
  -m MODEL, --model MODEL
                        model for embedding (default: sentence-transformers/sentence-t5-xl)
  -gpt GPT_MODEL, --gpt_model GPT_MODEL
                        model for topic representation and summary (default: None)
```

</details>
