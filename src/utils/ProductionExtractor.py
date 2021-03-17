from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
from stanza.server import CoreNLPClient
import numpy as np
import pandas as pd
import os
import pickle

class ProductionExtractor(AbstractFeatureExtractor):

    def __init__(self, config):
        '''
        Args:
            directories: a list of directory in which all documents will be used
                         to compute constituency.
        '''
        self.rules = {}
        if config.collectProductionFromCorpus:
            with CoreNLPClient(
                annotators=['tokenize','ssplit','pos','lemma','ner', 'parse'],
                timeout=30000,
                memory='16G') as client:

                for directory in config.directories:
                    for file in os.listdir(directory):
                        filePath = os.path.join(directory, file)
                        with open(filePath) as f:
                            self._countRewriteRules(f.read(), client)

    def featureName(self) -> list:
        return [keyValue[0] for keyValue in sorted(self.rules.items(), key=lambda x : x[1])]

    def extract(self, data : list) -> np.array:

        with CoreNLPClient(
            annotators=['parse'],
            timeout=30000,
            memory='16G') as client:
            out = []
            for text in data:
                annotate = client.annotate(text)
                array = np.zeros(len(self.rules))
                numberOfProduction = 0
                for sentence in annotate.sentence:
                    numberOfProduction += self._extractProductionsAndCountLeaves(sentence.parseTree, array)

                array = array/numberOfProduction
                out.append(array)

            return np.array(out)

    def save(self, pathToProduction):
        with open(pathToProduction, 'wb') as f:
            pickle.dump(self.rules, pathToProduction)

    def load(self, pathToProduction):
        with open(pathToProduction, 'rb') as f:
            self.rules = pickle.load(f)

    def _countRewriteRules(self, text, client):

            annotate = client.annotate(text)
            for sentence in annotate.sentence:
                self._addRules(sentence.parseTree)
            
    def _addRules(self, tree):

        if not self._lastInnerNode(tree):

            production = self._getProduction(tree)
            if production not in self.rules:
                self.rules[production] = len(self.rules)

            for child in tree.child:
                self._addRules(child)

    def _lastInnerNode(self, tree):

        lastInner = True
        for child in tree.child:
            if child.child:
                lastInner = False
                break
        return lastInner

    def _extractProductionsAndCountLeaves(self, tree, array) -> int:
        ''' count the number of each production and the total number of production used in the tree.
        
        Return:
            number of production in this tree
        '''
        if self._lastInnerNode(tree):
            return 0
        else:
            production = self._getProduction(tree)
            if production in self.rules:
                index = self.rules[production]
                array[index] += 1

            numberOfProduction = 1
            for child in tree.child:
                numberOfProduction += self._extractProductionsAndCountLeaves(child, array)
            return numberOfProduction

    def _getProduction(self, tree):

        return f"{tree.value} -> {' '.join([child.value for child in tree.child])}"