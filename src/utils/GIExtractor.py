from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
import numpy as np
import pandas as pd


class GIExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):

        self.table = pd.read_excel(config.pathToGI, dtype=str).drop(columns=['Source', 'Othtags', 'Defined'])
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

    def extract(self, data : list) -> np.array:
        wnl = WordNetLemmatizer()
        
        out = []
        for text in data:
            count = np.zeros(len(self.table.columns) - 1)
            for sent in sent_tokenize(text):
                words = word_tokenize(sent)
                for word in words:
                    word = word.lower()
                    for pos in ['v', 'a', 'n']:
                        lemmatizedWord = wnl.lemmatize(word, pos)
                        if lemmatizedWord in self.words:
                            for categoryIndex in self.wordCategoriesIndex[lemmatizedWord]:
                                count[categoryIndex] += 1
                            break

            out.append(np.array(count))
        return np.array(out)