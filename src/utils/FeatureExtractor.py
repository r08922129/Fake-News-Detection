from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np
import importlib
import pickle

class FeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, config):
        
        self.config = config
        self.extractors = []
        if config.init:
            for moduleName in config.modules:
                module = importlib.import_module("{}.{}".format(config.sourceBase, moduleName))
                self.extractors.append(getattr(module, moduleName)(config))
        
    def featureName(self) -> list:
        names = []
        for extractor in self.extractors:
            names.extend(extractor.featureName())

        return  names

    def extractorName(self):
        return [extractor.__class__.__name__ for extractor in self.extractors]

    def extract(self, data : list) -> np.array:
        '''Extract features of input text
        
        Args:
            data : a list of string
        Return:
            a numpy array with shape (n, m)
            where n is the number of samples and m is the dimentsion of features 
        '''
        features = [extractor.extract(data) for extractor in self.extractors]
        return np.concatenate(features, axis=1)

    def extractorIndices(self):
        '''
        Return: a dictionary with name of extractor as key and ndarray of indices as value
        '''
        out = {}
        index = 0
        for extractor in self.extractors:
            length = len(extractor.featureName())
            out[extractor.__class__.__name__] = np.arange(index, index+length)
            index += length
        return out

    def save(self, path):
        check_point ={
            'config' : self.config,
            'extractors' : self.extractors
        }
        with open(path , 'wb') as f:
            pickle.dump(check_point, f)

    def load(self, path):
        with open(path, 'rb') as f:
            check_point = pickle.load(f)
            self.config = check_point['config']
            self.extractors = check_point['extractors']

    def drop(self, extractorName):

        for i, extractor in enumerate(self.extractors):
            if extractor.__class__.__name__ == extractorName:
                print("Dropped {}.".format(extractor.__class__.__name__))
                self.extractors.pop(i)





