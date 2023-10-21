import string
import pandas as pd
import random
from sklearn.metrics import accuracy_score
import numpy as np

POS_LABEL = [
    "NOUN", "ADP", "DET", "PUNCT", "VERB", "PROPN", "ADJ", "PRON", "ADV",
    "AUX", "CCONJ", "NUM", "X", "SCONJ", "SYM", "INTJ"
]





class NaiveClassifier:
    def __init__(self) -> None:
        pass

    def predict(self, data: pd.DataFrame):

        assert 'FORM' in data.columns, 'Column FORM is missing from data'
        def naive_rules(word):
            word = word.lower()  # Convert word to lowercase for consistent rule checking
            if word.endswith(('tion', 'sion', 'ment')) or word in ['ville', 'année', 'pays', 'nom', 'jour', 'mois', 'homme', 'femme', 'vie', 'enfant']:
                return 'NOUN'
            if word.endswith(('er', 'ir', 're')) or word in ['être', 'avoir', 'est', 'était', 'été', 'fut', 'étant']:
                return 'VERB'
            if word in ['être', 'avoir', 'est', 'était', 'été', 'fut', 'étant']:
                return 'AUX'
            if word.endswith(('eux', 'euse')) or word in ['premier', 'première', 'français', 'française', 'nouveau', 'nouvelle', 'grand', 'grande', 'petit', 'petite']:
                return 'ADJ'
            if word.endswith('ment') or word in ['très', 'bien', 'aussi', 'toujours', 'surtout', 'encore', 'trop', 'jamais', 'vraiment']:
                return 'ADV'
            if word in ['le', 'la', 'les', 'l', 'un', 'une', 'des']:
                return 'DET'
            if word in ['il', 'elle', 'nous', 'vous', 'ils', 'elles', 'lui', 'on', 'je', 'me', 'ma', 'mon', 'leur', 'leurs']:
                return 'PRON'
            if word in ['et', 'ou', 'mais', 'donc']:
                return 'CCONJ'
            if word in ['que', 'quand', 'si', 'lorsque']:
                return 'SCONJ'
            if word in ['de', 'à', 'en', 'sur', 'avec', 'dans', 'par', 'pour', 'sans', 'sous', 'contre', 'après', 'avant', 'vers', 'chez']:
                return 'ADP'
            if word.isdigit() or word in ['un', 'deux', 'trois', 'deuxième', 'troisième']:
                return 'NUM'
            if word[0].isupper() and word not in ['et', 'ou', 'mais', 'donc']:
                return 'PROPN'
            if word in [',', '.', '(', ')', ':', ';', '«', '»', '!', '?']:
                return 'PUNCT'
            if word in ['%', '€']:
                return 'SYM'
            if word in ['oh', 'ah', 'eh']:
                return 'INTJ'
            return 'NOUN'
        
        return data.FORM.apply(naive_rules).tolist()
    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class RandomClassifier:

    def __init__(self, random_seed=0) -> None:
        self.random_seed = random_seed
        self.label = POS_LABEL
    
    def predict(self, data: pd.DataFrame):
        random.seed(self.random_seed)
        return [random.choice(self.label) for _ in range(len(data))]
    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class StratifiedClassifier: 
    def __init__(self) -> None :
        pass
    def predict(self, data: pd.DataFrame):
        valuecounts= data['UPOS'].value_counts(normalize=True).to_dict()
        probabilities = list(valuecounts.values())
        labels = list(valuecounts.keys())
        N=len(data)
        
        return np.random.choice(POS_LABEL, size=N, p=probabilities)
    def evaluate(self, data: pd.DataFrame):
        y_true = data['UPOS'].tolist()
        y_pred = self.predict(data)
        
        assert len(y_true) == len(y_pred), "y_true and y_pred do not have the same length"
        
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        return metrics

class MostCommonPOSClassifier:
    def __init__(self) -> None:
        self.most_common_pos = None

    def predict(self, data: pd.DataFrame):
        self.most_common_pos = data['UPOS'].value_counts().idxmax()    
        return [self.most_common_pos] * len(data)
    
    def evaluate(self, data: pd.DataFrame):
        y_true = data['UPOS'].tolist()
        y_pred = self.predict(data)
        assert len(y_true) == len(y_pred), "y_true and y_pred do not have the same length"
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        return metrics

class WordDistributedRandomClassifier:
    def __init__(self):
        self.word_distributions = {}
        self.global_distribution = None

    def fit(self, data: pd.DataFrame):
        #on train_data
        # what's the POS distribution on each word 
        self.global_distribution = data['UPOS'].value_counts(normalize=True).to_dict()
        for word, group in data.groupby('FORM'):
            self.word_distributions[word] = group['UPOS'].value_counts(normalize=True).to_dict()

    def predict(self, data: pd.DataFrame):
        predictions = []
        for word in data['FORM']:
            if word in self.word_distributions:
                pos_tags = list(self.word_distributions[word].keys())
                probabilities = list(self.word_distributions[word].values())
                predictions.append(np.random.choice(pos_tags, p=probabilities))
            else:
                pos_tags = list(self.global_distribution.keys())
                probabilities = list(self.global_distribution.values())
                predictions.append(np.random.choice(pos_tags, p=probabilities))
        return predictions

    def evaluate(self, data: pd.DataFrame):
        y_true = data['UPOS'].tolist()
        y_pred = self.predict(data)
        return accuracy_score(y_true, y_pred)



