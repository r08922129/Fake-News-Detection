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
          'FleschKincaidGradeLevel',
          'AutomatedReadabilityIndex',
          'Coleman-Liau',
          'FleschReadingEase',
          'GunningFogIndex',
          'LIX',
          'SMOGIndex',
          'RIX',
          'DaleChallIndex'
          ]
    
    def extract(self, data : list) -> np.array:

        array = []
        for text in data:
            array.append(list(readability.getmeasures(text, lang='en')['readability grades'].values()))

        return np.array(array)