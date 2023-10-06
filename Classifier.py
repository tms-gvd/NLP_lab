import pandas as pd
import matplotlib.pyplot as plt
import string

class ConLLUParser:

    def __init__(self, path, verbose=False) -> None:
        # open .conllu file and split each line and remove the '\n' character at the end
        with open(path) as f:
            raw_lines = f.read().splitlines()
            #splitting lines from the conlu parser 
        
        # retrieve the name of the columns
        self.col_names = [
            'ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'
        ]
        
        # all_words stores each word of the corpus, identified by sent_id (id of the sentence in the corpus) and the ID (position of the word in the sentence)
        # all_contractions stores the contractions, e.g. 'du' -> 'de le' ; 'didn't' -> 'did n't'
        # all_sentences stores each sentence of the corpus, identified by sent_id and computes the length of the sentence
        all_words, all_contractions, all_sentences = [], [], []
        
        # temporary variables
        temp, count, n_exceptions = [], 0, 0

        # treat the corpus line by line using pop function
        while len(raw_lines) > 0:
            
            line = raw_lines.pop(0)
            if len(line) > 0:
                splits = line.split('\t')
            
            # the sentences always start by a line giving the sent_id "# sent_id = ..."
            if line.startswith('# sent_id = '):
                id = line.replace('# sent_id = ', '')
                temp.append(id)
                
            
            # then another one gives the sentence "# text = ..."
            elif line.startswith('# text = '):
                text = line.replace('# text = ', '')
                temp.append(text)
            
            # a sentence is ended by a blank line
            elif line == '':
                if count > 0:
                    temp.append(count)
                    all_sentences.append(temp)
                count = 0
                temp = []

            # print all lines that don't start by a digit as their are not useful
            # count them for later sanity checks
            elif not line[0].isdigit():
                if verbose:
                    print(f'IS OMITTED: {line}')
                n_exceptions += 1

            # the lines that start by a number contain the words and their characteristics
            elif splits[0].isdigit():
                assert len(splits) == len(self.col_names)
                all_words.append([id] + splits)
                count += 1

            # if the start of a line contains a dash, then it represents a contradiction
            elif '-' in splits[0] or '.' in splits[0]:
                all_contractions.append([id] + splits[:2])
            
            else:
                raise ValueError(f'Cannot handle this line: {line}')
            

        self.sentences = pd.DataFrame(all_sentences, columns=['sent_id', 'text', 'length'])
        self.words = pd.DataFrame(all_words, columns=['sent_id'] + self.col_names)
        self.contradictions = pd.DataFrame(all_contractions, columns=['sent_id', 'ID', 'FORM'])
            


if __name__=="__main__":

    verbose = False
    fr_train = ConLLUParser('fr_gsd-ud-train.conllu', verbose)
    fr_dev = ConLLUParser('fr_gsd-ud-dev.conllu', verbose)
    fr_test = ConLLUParser('fr_gsd-ud-test.conllu', verbose)

