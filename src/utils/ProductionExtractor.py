from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
from stanza.server import CoreNLPClient
import numpy as np
import pandas as pd
import os
from nltk import sent_tokenize
import pickle
from tqdm import tqdm
class ProductionExtractor(AbstractFeatureExtractor):

    def __init__(self, config):

        self.rules = {}
        if config.collectProductionFromCorpus:
            with CoreNLPClient(
                annotators=['tokenize','ssplit', 'parse'],
                timeout=30000,
                memory='16G') as client:

                for filePath in config.pathesToFiles:
                    with open(filePath) as f:
                        text = [line.strip() for line in f.readlines()]
                        for paragraph in text:
                            for sent in sent_tokenize(paragraph):
                                self._countRewriteRules(sent, client, filePath)

    def featureName(self) -> list:
        return [keyValue[0] for keyValue in sorted(self.rules.items(), key=lambda x : x[1])]

    def extract(self, data : list) -> np.array:

        with CoreNLPClient(
            annotators=['tokenize','ssplit', 'parse'],
            timeout=30000,
            memory='16G') as client:
            out = []
            for text in tqdm(data):
                array = np.zeros(len(self.rules))
                numberOfProduction = 0
                for sent in sent_tokenize(text):
                    try:
                        annotate = client.annotate(sent)
                        for sentence in annotate.sentence:
                            numberOfProduction += self._extractProductionsAndCountLeaves(sentence.parseTree, array)
                    except:
                        continue
                array = array/numberOfProduction
                out.append(array)

            return np.array(out)

    def save(self, pathToProduction):
        with open(pathToProduction, 'wb') as f:
            pickle.dump(self.rules, pathToProduction)

    def load(self, pathToProduction):
        with open(pathToProduction, 'rb') as f:
            self.rules = pickle.load(f)

    def _countRewriteRules(self, text, client, filePath):
        try:
            annotate = client.annotate(text)
            for sentence in annotate.sentence:
                self._addRules(sentence.parseTree)
        except:
            print(f"File {filePath} too long.")

            
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