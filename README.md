ml-preprocess-tools
===

### Dependency

```
$ sudo apt-get install mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8
```

### Install

```
$ pip install git+https://github.com/altescy/ml-preprocess-tools
```

You will need to download models and languages for spaCy and NLTK.

```
$ python -m spacy download en
$ python -c "import nltk;nltk.download('wordnet')"
```
