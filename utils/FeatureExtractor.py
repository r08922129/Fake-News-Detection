from .AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np

class FeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, config):
        
        self.config = config
        self.extractors = []
        
    def featureName(self) -> list:
        
        return NotImplemented
    def extract(self, text) -> np.array:

        return NotImplemented




