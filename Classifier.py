import string
import pandas as pd

POS_LABEL = [
    "NOUN",
    "ADP",
    "DET",
    "PUNCT",
    "VERB",
    "PROPN",
    "ADJ",
    "PRON",
    "ADV",
    "AUX",
    "CCONJ",
    "NUM",
    "X",
    "SCONJ",
    "SYM",
    "INTJ"
]

class NaiveClassifier:

    def __init__(self) -> None:
        pass

    def predict(self, data:pd.DataFrame):

        assert 'FORM' in data.columns, 'Column FORM is missing from data'
        
        def naive_rules(word):
            if word.endswith('s'):
                return 'VERB'
            elif word[0].isupper():
                return 'PROPN'
            elif word.isdigit():
                return 'NUM'
            elif word.lower().startswith('qu'):
                return 'SCONJ'
            elif all(i in string.punctuation for i in word):
                return 'PUNCT'
            else:
                return 'NOUN'
        return data.FORM.apply(naive_rules).tolist()


class RandomClassifier:

    def __init__(self, random_seed=0) -> None:
        self.random_seed = random_seed
        self.label = POS_LABEL
    
    def predict(self, data):
        pass