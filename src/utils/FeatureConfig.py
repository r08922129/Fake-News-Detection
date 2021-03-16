import os
from stanza.server import CoreNLPClient

class FeatureConfig(object):
    
    def __init__(
        self,
        pathToDataset,
        sourceBase = 'src.utils',
        bow = True,
        vocabSize = 5000,
        gi = True,
        pathToGI = None,
        pos = True,
        collectPosFromCorpus = False,
        production = True,
        collectProductionFromCorpus = False,
        readability = True,
        quantity = True,
        sentiment = True,
        punctutation = False,
    ):
        self.directories = []
        for directory in os.listdir(pathToDataset):
            path = os.path.join(pathToDataset, directory)
            if os.path.isdir(path):
                self.directories.append(path)
        self.vocabSize = vocabSize
        self.sourceBase = sourceBase
        self.modules = []

        self.collectPosFromCorpus = collectPosFromCorpus
        self.collectProductionFromCorpus = collectProductionFromCorpus
        if gi:
            if not pathToGI:
                raise Exception("Path to GI dictionary should be assigned.")
            if not os.path.isfile(pathToGI):
                raise Exception("Path to GI dictionary is invalid.")
            self.modules.append("GIExtractor")
            self.pathToGI = pathToGI
        if bow:
            self.modules.append("BOWExtractor")
        if pos:
            self.modules.append("PosExtractor")
        if production:
            self.modules.append("ProductionExtractor")
        if readability:
            self.modules.append("ReadabilityExtractor")
        if quantity:
            self.modules.append("QuantityExtractor")
        if sentiment:
            self.modules.append("SentimentExtractor")
        if punctutation:
            self.modules.append("PunctuationExtractor")