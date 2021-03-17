from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np
import nltk
import os
import pickle

class PosExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):
        
        self.posTag = {}
        self.config = config

        if config.collectPosFromCorpus:
            for filePath in config.pathesToFiles:
                with open(filePath) as f:
                    for paragraph in f.readlines():
                        self._countPos(paragraph)

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
                    if pos in self.posTag:
                        index = self.posTag[pos]
                        array[index] += 1
                if self.config.posBigram:
                    for i in range(len(tagging)-1):
                        bigram =  '{} {}'.format(tagging[i][1], tagging[i+1][1])
                        if bigram in self.posTag:
                            index = self.posTag[bigram]
                            array[index] += 1
                if self.config.posTrigram:
                    for i in range(len(tagging)-2):
                        trigram =  '{} {} {}'.format(tagging[i][1], tagging[i+1][1], tagging[i+2][1])
                        if trigram in self.posTag:
                            index = self.posTag[trigram]
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
            # unigram
            for word, pos in tagging:
                if pos not in self.posTag:
                    self.posTag[pos] = len(self.posTag)
            # bigram
            if self.config.posBigram:
                for i in range(len(tagging)-1):
                    bigram =  '{} {}'.format(tagging[i][1], tagging[i+1][1])
                    if bigram not in self.posTag:
                        self.posTag[bigram] = len(self.posTag)
            # trigram
            if self.config.posTrigram:
                for i in range(len(tagging)-2):
                    trigram =  '{} {} {}'.format(tagging[i][1], tagging[i+1][1], tagging[i+2][1])
                    if trigram not in self.posTag:
                        self.posTag[trigram] = len(self.posTag)