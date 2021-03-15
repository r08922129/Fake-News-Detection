import abc
import numpy as np

class AbstractFeatureExtractor(abc.ABC):
    
    @abc.abstractmethod
    def featureName(self) -> list:
        '''
        Return:
            keys: feature name of each dimension.
        '''
        return NotImplemented

    @abc.abstractmethod
    def extract(self, text: str) -> np.array:
        '''To extract linguistic features
        
        Args:
            a text
            
        Return:
            array: a numpy array with n dimension determined by the number of feature specified in config

        '''
        return NotImplemented