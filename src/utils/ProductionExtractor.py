from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
from stanza.server import CoreNLPClient
import numpy as np
import pandas as pd
import os
from nltk import sent_tokenize
import pickle
from tqdm import tqdm
import json
from src.utils.UtilsOfTree import core_nlp_tree_to_json, collect_productions_from_json_tree
class ProductionExtractor(AbstractFeatureExtractor):

    def __init__(self, config):

        self.rules = {}
        if config.collectProductionFromCorpus:
            for filePath in config.pathesToTree:
                with open(filePath) as f:
                    trees = [json.loads(line) for line in f.readlines()]
                    for tree in trees:
                        self._addRules(tree)

    def featureName(self) -> list:
        return [keyValue[0] for keyValue in sorted(self.rules.items(), key=lambda x : x[1])]

    def extract(self, data) -> np.array:

        with CoreNLPClient(
            annotators=['tokenize','ssplit', 'parse'],
            timeout=30000,
            memory='16G') as client:
            out = []
            for text in tqdm(data):
                array = np.zeros(len(self.rules))
                numberOfProduction = 0
                for line in text.split("\n"):
                    for sent in sent_tokenize(line):
                        try:
                            annotate = client.annotate(sent)
                            for sentence in annotate.sentence:
                                tree = core_nlp_tree_to_json(sentence.parseTree)
                                numberOfProduction += self._accumulateProduction(sentence.parseTree, array)
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

    def _countRewriteRules(self, tree):
        productions = collect_productions_from_json_tree(tree)
        for production in productions:
            if production not in self.rules:
                self.rules[production] = len(self.rules)

    def _accumulateProduction(self, parseTree, array) -> int:
        ''' count the number of each production and the total number of production used in the tree.
        
        Return:
            number of production in this tree
        '''

        productions = collect_productions_from_json_tree(tree)
        for production in productions:
            if production in self.rules:
                index = self.rules[production]
                array[index] += 1
        return len(productions)