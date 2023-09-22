import string

class NaiveClassifier:

    def __init__(self, train, dev, test) -> None:
        self.train = train
        self.dev = dev
        self.test = test

    def fit(self, train):
        
        def naive_rules(word):
            if word.endswith('s'):
                return 'VERB'
            if word[0].isupper():
                return 'PROPN'
            if word.isdigit():
                return 'NUM'
            if word.lower().startswith('qu'):
                return 'SCONJ'
            if (i in string.punctuation for i in word):
                return 'PUNCT'
        
        train['pred'] = train.FORM.apply(naive_rules)