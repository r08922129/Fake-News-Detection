from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np
import nltk
import os
import pickle

class PosExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):
        
        self.posTag = {}
        
        if config.collectPosFromCorpus:
            for directory in config.directories:
                for file in os.listdir(directory):
                    filePath = os.path.join(directory, file)
                    with open(filePath) as f:
                        self._countPos(f.read())

    def featureName(self) -> list:
        return [keyValue[0] for keyValue in sorted(self.posTag.items(), key=lambda x : x[1])]

    def extract(self, data : list) -> np.array:

        out = []
        for text in data:
            array = np.zeros(len(self.posTag))
            numberOfWords = 0
            for sentence in nltk.sent_tokenize(text):
                words = nltk.word_tokenize(sentence)
                numberOfWords += len(words)
                tagging = nltk.pos_tag(words)
                for word, pos in tagging:
                    index = self.posTag[pos]
                    array[index] += 1
            array = array/numberOfWords
            out.append(array)
        return np.array(out)

    def save(self, pathToPos):
        with open(pathToPos, 'wb') as f:
            pickle.dump(self.posTag, f)

    def load(self, pathToPos):
        with open(pathToPos, 'rb') as f:
            self.posTag = pickle.load(f)
 
    def _countPos(self, text):

        for sentence in nltk.sent_tokenize(text):
            words = nltk.word_tokenize(sentence)
            tagging = nltk.pos_tag(words)
            for word, pos in tagging:
                if pos not in self.posTag:
                    self.posTag[pos] = len(self.posTag)
             