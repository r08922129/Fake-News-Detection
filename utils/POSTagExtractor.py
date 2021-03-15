from .AbstractFeatureExtractor import AbstractFeatureExtractor
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
import numpy as np
import pandas as pd

# nltk.download('averaged_perceptron_tagger')
class POSExtractor(AbstractFeatureExtractor):
    
    def __init__(self, pathToGI):

        self.table = pd.read_excel(pathToGI, dtype=str).drop(columns=['Source', 'Othtags', 'Defined'])
        self.words = set(self.table['Entry'].map(lambda w : w.lower()))
        self.wordCategoriesIndex = {
            word : [] for word in self.words
        }

        for index, category in enumerate(self.table.columns[1:]):
            wordsIndices = self.table[category].map(lambda v : type(v) == str).to_numpy().nonzero()[0]
            for word in self.table['Entry'][wordsIndices]:
                self.wordCategoriesIndex[word.lower()].append(index)
            
    def featureName(self) -> list:
        return self.table.columns[1:]

    def extract(self, text: str) -> np.array:
        wnl = WordNetLemmatizer()
        
        count = [0] * (len(self.table.columns) - 1)
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                for pos in ['v', 'a', 'n']:
                    if wnl.lemmatize(word, pos) in self.words:
                        for categoryIndex in self.wordCategoriesIndex[word]:
                            count[categoryIndex] += 1
                        break

        return np.array(count)
                