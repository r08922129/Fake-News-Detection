from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize, sent_tokenize
import re
import readability

class ReadabilityExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):
        pass
    
    def featureName(self) -> list:
        
        return [
          'FleschReadingEaseIndex',
         'FleschKincaidGradeLevel',
          'AutomatedReadabilityIndex',
          'GunningFoxIndex',
          'ColemanLiauIndex',
          '#Words',
          '#Syllables',
          '#Polysyllables',
          '#Chars',
          '#ComplexWords'
          ]
    
    def extract(self, data : list) -> np.array:

        array = list(readability.getmeasures("test", lang='en')['readability grades'].values())

        return np.array(array)