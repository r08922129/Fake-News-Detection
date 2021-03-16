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
    def extract(self, data : list) -> np.array:
        '''Extract features of input text
        
        Args:
            data : a list of string
        Return:
            a numpy array with shape (n, m)
            where n is the number of samples and m is the dimentsion of features 
        '''
        return NotImplemented